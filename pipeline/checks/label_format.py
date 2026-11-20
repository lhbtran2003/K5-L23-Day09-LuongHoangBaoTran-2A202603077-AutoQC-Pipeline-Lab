"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def _bad_format(stem, line_no, text):
    try:
        return parse_yolo_line(text), None
    except ValueError as exc:
        return None, Issue("label_format", ERROR, "BAD_FORMAT", stem, str(exc), line=line_no)


def _bad_class_id(stem, line_no, box, class_count):
    class_id = box[0]
    if not 0 <= class_id < class_count:
        return Issue("label_format", ERROR, "BAD_CLASS_ID", stem,
                     f"id lớp {class_id} nằm ngoài khoảng 0..{class_count - 1}", line=line_no)
    return None


def _box_zero(stem, line_no, box):
    if box[3] <= 0 or box[4] <= 0:
        return Issue("label_format", ERROR, "BOX_ZERO", stem,
                     "chiều rộng và chiều cao box phải lớn hơn 0", line=line_no)
    return None


def _box_out_of_image(stem, line_no, box, tolerance):
    _, cx, cy, width, height = box
    left, top = cx - width / 2, cy - height / 2
    right, bottom = cx + width / 2, cy + height / 2
    overflow = max(0.0, -left, -top, right - 1.0, bottom - 1.0)
    if overflow > tolerance:
        return Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem,
                     f"box tràn khỏi ảnh {overflow:.6g}", line=line_no,
                     extra={"overflow": overflow})
    return None


def _dup_box(stem, line_no, box, previous_boxes, iou_threshold):
    class_id = box[0]
    coords = box[1:]
    if any(prev_class == class_id and iou_xywh(coords, prev_coords) > iou_threshold
           for prev_class, prev_coords in previous_boxes):
        return Issue("label_format", ERROR, "DUP_BOX", stem,
                     "box gần trùng với box cùng lớp ở dòng trước", line=line_no)
    return None


def _box_tiny(ds, stem, line_no, box, min_box_px):
    size = ds.image_size(stem)
    if size is None:
        return None
    image_width, image_height = size
    short_side_px = min(box[3] * image_width, box[4] * image_height)
    if short_side_px < min_box_px:
        return Issue("label_format", WARN, "BOX_TINY", stem,
                     f"cạnh ngắn của box chỉ {short_side_px:.6g}px", line=line_no,
                     extra={"short_side_px": short_side_px})
    return None


def _box_huge(stem, line_no, box, max_box_frac):
    area_fraction = box[3] * box[4]
    if area_fraction > max_box_frac:
        return Issue("label_format", WARN, "BOX_HUGE", stem,
                     f"diện tích box chiếm {area_fraction:.6g} ảnh", line=line_no,
                     extra={"area_fraction": area_fraction})
    return None


def run(ds, cfg):
    settings = cfg["label_format"]
    issues = []

    for label_file in ds.label_files():
        stem = label_file.stem
        previous_boxes = []
        for line_no, text in ds.read_label_lines(stem) or []:
            box, issue = _bad_format(stem, line_no, text)
            if issue is not None:
                issues.append(issue)
                continue

            issue = _bad_class_id(stem, line_no, box, len(ds.classes))
            if issue is not None:
                issues.append(issue)
                continue

            issue = _box_zero(stem, line_no, box)
            if issue is not None:
                issues.append(issue)
                continue

            checks = (
                _box_out_of_image(stem, line_no, box, settings["overflow_tol"]),
                _dup_box(stem, line_no, box, previous_boxes, settings["dup_box_iou"]),
                _box_tiny(ds, stem, line_no, box, settings["min_box_px"]),
                _box_huge(stem, line_no, box, settings["max_box_frac"]),
            )
            issues.extend(issue for issue in checks if issue is not None)
            previous_boxes.append((box[0], box[1:]))

    return issues
