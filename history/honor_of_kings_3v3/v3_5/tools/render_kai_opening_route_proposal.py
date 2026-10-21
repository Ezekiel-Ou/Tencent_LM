from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "开发指南" / "images" / "项目简介" / "长平攻防战_d79fa4eeb9.png"
OUTPUT = ROOT / "docs" / "debug" / "kai_opening_route_coordinate_proposal.png"

FONT_CJK = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_MONO = Path(r"C:\Windows\Fonts\consola.ttf")


def font(path, size):
    return ImageFont.truetype(str(path), size)


def arrow(draw, points, color, width=9, dash=False):
    outline = width + 5
    if dash:
        for start, end in zip(points, points[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 0:
                continue
            ux, uy = dx / length, dy / length
            cursor = 0.0
            while cursor < length:
                a = cursor
                b = min(cursor + 24, length)
                p0 = (start[0] + ux * a, start[1] + uy * a)
                p1 = (start[0] + ux * b, start[1] + uy * b)
                draw.line((p0, p1), fill=(255, 255, 255, 230), width=outline)
                draw.line((p0, p1), fill=color, width=width)
                cursor += 38
    else:
        draw.line(points, fill=(255, 255, 255, 235), width=outline, joint="curve")
        draw.line(points, fill=color, width=width, joint="curve")

    start, end = points[-2], points[-1]
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 22
    left = (
        end[0] - size * math.cos(angle - math.pi / 6),
        end[1] - size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - size * math.cos(angle + math.pi / 6),
        end[1] - size * math.sin(angle + math.pi / 6),
    )
    draw.polygon((end, left, right), fill=(255, 255, 255, 235))
    inset = 5
    left_inner = (
        end[0] - (size - inset) * math.cos(angle - math.pi / 6),
        end[1] - (size - inset) * math.sin(angle - math.pi / 6),
    )
    right_inner = (
        end[0] - (size - inset) * math.cos(angle + math.pi / 6),
        end[1] - (size - inset) * math.sin(angle + math.pi / 6),
    )
    draw.polygon((end, left_inner, right_inner), fill=color)


def marker(draw, number, position, color):
    x, y = position
    radius = 15
    draw.ellipse(
        (x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3),
        fill=(255, 255, 255, 245),
    )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
    )
    label = str(number)
    label_font = font(FONT_MONO, 20)
    box = draw.textbbox((0, 0), label, font=label_font)
    draw.text(
        (x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - 2),
        label,
        font=label_font,
        fill="white",
    )


def main():
    base = Image.open(SOURCE).convert("RGBA")
    canvas = Image.new("RGBA", (base.width, base.height + 230), (17, 24, 34, 255))
    canvas.alpha_composite(base, (0, 0))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    blue = (25, 150, 255, 255)
    cyan = (0, 224, 220, 255)
    red = (245, 52, 65, 255)
    orange = (255, 151, 28, 255)

    points = {
        1: ("风狼", (-14010, 27670), (644, 143), blue),
        2: ("鹿", (-25530, 23990), (534, 184), cyan),
        3: ("上野返回点", (-25000, 15000), (539, 257), red),
        4: ("上野主路入口", (-25000, 6115), (539, 336), red),
        5: ("下野主路入口", (-23000, 6115), (558, 348), red),
        6: ("先锋", (-23470, -11250), (555, 514), red),
        7: ("外侧弓手", (-29860, -24480), (493, 632), orange),
        8: ("内侧弓手", (-19840, -26940), (589, 650), orange),
        9: ("红像", (-20860, -17570), (579, 552), orange),
    }

    # Blue follows the fixed base -> upper entry -> inner point -> wolf route.
    # Deer is crossed only on the later cyan return leg, never as a wolf-route
    # waypoint. Numbered endpoints below are measured raw anchors.
    arrow(
        draw,
        [
            (175, 225),
            (305, 306),
            (430, 306),
            points[4][2],
            points[3][2],
            points[1][2],
        ],
        blue,
    )
    arrow(draw, [points[1][2], points[2][2]], cyan)
    arrow(
        draw,
        [
            points[2][2],
            points[3][2],
            points[4][2],
            points[5][2],
            points[6][2],
        ],
        red,
    )
    arrow(
        draw,
        [points[6][2], points[7][2], points[8][2], points[9][2]],
        orange,
    )

    for number, (_name, _raw, pixel, color) in points.items():
        marker(draw, number, pixel, color)

    title_font = font(FONT_CJK, 28)
    small_font = font(FONT_CJK, 18)
    draw.rounded_rectangle(
        (890, 26, 1550, 100),
        radius=14,
        fill=(7, 13, 22, 205),
        outline=(255, 255, 255, 120),
        width=2,
    )
    draw.text((914, 38), "凯固定开局路线坐标提案", font=title_font, fill="white")
    draw.text(
        (914, 73),
        "圆点为实测 raw 坐标；连线为待确认的规则路线",
        font=small_font,
        fill=(220, 229, 240, 255),
    )

    canvas = Image.alpha_composite(canvas, overlay)
    footer = ImageDraw.Draw(canvas)
    footer.rectangle((0, base.height, canvas.width, canvas.height), fill=(17, 24, 34, 255))
    footer.text(
        (42, base.height + 22),
        "顺序：基地 → 风狼 → 鹿 → 返回己方入口 → 先锋 → 外侧弓手 → 内侧弓手 → 红像",
        font=font(FONT_CJK, 25),
        fill="white",
    )

    columns = [(42, base.height + 72), (540, base.height + 72), (1035, base.height + 72)]
    rows = [
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
    ]
    body_font = font(FONT_CJK, 20)
    mono_font = font(FONT_MONO, 20)
    for column, numbers in zip(columns, rows):
        x, y = column
        for number in numbers:
            name, raw, _pixel, color = points[number]
            footer.ellipse((x, y + 2, x + 22, y + 24), fill=color)
            footer.text((x + 7, y + 1), str(number), font=font(FONT_MONO, 15), fill="white")
            footer.text((x + 34, y), name, font=body_font, fill=(235, 240, 247, 255))
            footer.text(
                (x + 175, y + 1),
                f"({raw[0]},{raw[1]})",
                font=mono_font,
                fill=(174, 201, 232, 255),
            )
            y += 42

    footer.text(
        (42, base.height + 204),
        "注：camp 1 原始坐标；camp 2 营地使用实测镜像侧锚点，不以理想整数镜像替代。",
        font=font(FONT_CJK, 16),
        fill=(166, 180, 198, 255),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUTPUT, quality=95)
    print(OUTPUT)


if __name__ == "__main__":
    main()
