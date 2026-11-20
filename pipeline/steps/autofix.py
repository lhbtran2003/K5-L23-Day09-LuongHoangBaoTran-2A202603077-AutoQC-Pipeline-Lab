"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    issues = [issue for issue in issues if issue.severity == ERROR]
    fixes = []

    # Xóa ảnh trước để các issue dòng nhãn cũ của ảnh đó không gây lỗi.
    remove_images = {
        issue.image
        for issue in issues
        if issue.code.startswith("DUP_IMAGE_") or issue.code in remove_codes
    }
    for stem in sorted(remove_images):
        image_path = ds.image_path(stem)
        label_path = ds.label_path(stem)
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)
        fixes.append({
            "image": stem,
            "line": None,
            "action": "remove_image",
            "before": "ảnh và file nhãn trong ds",
            "after": "đã xóa khỏi ds",
        })

    # Ảnh thiếu nhãn được chuyển cho người; không tạo file nhãn giả.
    missing_labels = sorted({issue.image for issue in issues if issue.code == "MISSING_LABEL_FILE"})
    for stem in missing_labels:
        if stem not in remove_images:
            fixes.append({
                "image": stem,
                "line": None,
                "action": "needs_label",
                "before": "(không có file)",
                "after": "chuyển người gán",
            })

    # Sửa data.yaml một lần dù có nhiều issue cùng báo tên lớp.
    if any(issue.code == "BAD_CLASS_NAME" for issue in issues):
        before = ds.yaml_path.read_text(encoding="utf-8") if ds.yaml_path.exists() else ""
        data = yaml.safe_load(before) or {}
        data["names"] = list(cfg["classes"])
        after = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        if before != after:
            ds.yaml_path.write_text(after, encoding="utf-8")
        fixes.append({
            "image": "data.yaml",
            "line": None,
            "action": "rename_classes",
            "before": before,
            "after": after,
        })

    # Gom issue theo file và số dòng gốc; mỗi file chỉ được ghi lại một lần.
    line_issues = defaultdict(list)
    for issue in issues:
        if issue.image in remove_images or issue.line is None:
            continue
        if issue.code in drop_codes or issue.code == "BOX_OUT_OF_IMAGE":
            line_issues[issue.image].append(issue)

    for stem in sorted(line_issues):
        path = ds.label_path(stem)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        original_lines = text.splitlines()
        issues_by_line = defaultdict(list)
        for issue in line_issues[stem]:
            issues_by_line[issue.line].append(issue)

        output_lines = []
        changed = False
        for line_no, before in enumerate(original_lines, 1):
            line_issues_here = issues_by_line.get(line_no, [])
            if not line_issues_here:
                output_lines.append(before)
                continue

            codes = {issue.code for issue in line_issues_here}
            if codes & drop_codes:
                changed = True
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": before,
                    "after": "đã xóa dòng",
                })
                continue

            # BOX_OUT_OF_IMAGE: cắt phần giao của box với vùng ảnh [0, 1].
            try:
                class_id, cx, cy, width, height = parse_yolo_line(before)
            except ValueError:
                output_lines.append(before)
                continue

            left = max(0.0, cx - width / 2)
            right = min(1.0, cx + width / 2)
            top = max(0.0, cy - height / 2)
            bottom = min(1.0, cy + height / 2)
            if right <= left or bottom <= top:
                changed = True
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": before,
                    "after": "đã xóa box nằm ngoài ảnh",
                })
                continue

            clipped = (
                class_id,
                (left + right) / 2,
                (top + bottom) / 2,
                right - left,
                bottom - top,
            )
            after = " ".join(f"{value:.8g}" for value in clipped)
            output_lines.append(after)
            changed = changed or after != before
            if after != before:
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "clip_box",
                    "before": before,
                    "after": after,
                })

        if changed:
            result = "\n".join(output_lines)
            if text.endswith(("\n", "\r")):
                result += "\n"
            path.write_text(result, encoding="utf-8")

    return fixes
