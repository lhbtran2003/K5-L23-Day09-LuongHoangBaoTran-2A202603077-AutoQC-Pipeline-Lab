"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN  # noqa: F401


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    image_stems = set(image_stems)
    reviewed = set(reviewed)
    n_images = len(image_stems)

    # ERROR là lỗi bắt buộc: luôn được tính, kể cả khi ảnh đã được review.
    n_errors = sum(issue.severity == ERROR for issue in issues)

    # Cảnh báo được tính theo ảnh duy nhất, chỉ trong dataset và chưa được review.
    warn_images = {
        issue.image
        for issue in issues
        if issue.severity == WARN
        and issue.image in image_stems
        and issue.image not in reviewed
    }
    n_warn_images = len(warn_images)
    warn_ratio = n_warn_images / n_images if n_images else 0.0
    hold_threshold = cfg["gate"]["warn_ratio_hold"]

    if n_errors:
        decision = "Rework"
        reasons = [
            f"Rework: còn {n_errors} lỗi bắt buộc (severity ERROR); lỗi bắt buộc không được bỏ qua khi review.",
            f"Có cảnh báo trên {n_warn_images}/{n_images} ảnh chưa review (tỷ lệ {warn_ratio:.4%}; ngưỡng Hold {hold_threshold:.4%}).",
        ]
    elif warn_ratio >= hold_threshold:
        decision = "Hold"
        reasons = [
            f"Hold: {n_warn_images}/{n_images} ảnh chưa review còn cảnh báo (tỷ lệ {warn_ratio:.4%}), đạt/vượt ngưỡng {hold_threshold:.4%}.",
        ]
    else:
        decision = "Release"
        reasons = [
            f"Release: không còn lỗi bắt buộc; {n_warn_images}/{n_images} ảnh chưa review có cảnh báo (tỷ lệ {warn_ratio:.4%}), dưới ngưỡng Hold {hold_threshold:.4%}.",
        ]

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": round(warn_ratio, 4),
        "reasons": reasons,
    }
