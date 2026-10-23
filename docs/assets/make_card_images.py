"""The landing page's card pictures: a planet of each kind crossing a
limb-darkened star. Writes docs/_static/cards/*.svg.

    python docs/assets/make_card_images.py

Transparent backgrounds, colours that read on light and dark pages alike
(the cards fade them in under their text: custom.css).
"""
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "_static" / "cards"

W, H = 200, 120
SX, SY, SR = 128, 60, 54          # the star
CHORD = 72                        # the planet's path (y)
PLANET = "#1f2633"
GHOST = "#7d8796"
RING = "#a08f72"


def star(spots=""):
    return f"""<defs>
  <radialGradient id="limb" cx="50%" cy="50%" r="50%">
    <stop offset="0%" stop-color="#fff7da"/>
    <stop offset="55%" stop-color="#ffe08a"/>
    <stop offset="85%" stop-color="#ffbe55"/>
    <stop offset="100%" stop-color="#f08a2c"/>
  </radialGradient>
  <radialGradient id="glow" cx="50%" cy="50%" r="50%">
    <stop offset="80%" stop-color="#ffb347" stop-opacity="0.35"/>
    <stop offset="100%" stop-color="#ffb347" stop-opacity="0"/>
  </radialGradient>
</defs>
<circle cx="{SX}" cy="{SY}" r="{SR + 8}" fill="url(#glow)"/>
<circle cx="{SX}" cy="{SY}" r="{SR}" fill="url(#limb)"/>
{spots}
<line x1="18" y1="{CHORD}" x2="196" y2="{CHORD}" stroke="{GHOST}" stroke-width="1"
      stroke-dasharray="3 4" opacity="0.6"/>"""


def svg(body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}">\n{body}\n</svg>\n')


def track(shape, xs=(38, 64, 92, 118)):
    """Faded earlier positions along the chord, then the planet itself."""
    out = []
    for i, x in enumerate(xs):
        last = i == len(xs) - 1
        fill = PLANET if last else GHOST
        op = 1.0 if last else 0.18 + 0.14 * i
        out.append(shape(x, fill, op, last))
    return "\n".join(out)


def sphere(x, fill, op, last):
    return f'<circle cx="{x}" cy="{CHORD}" r="11" fill="{fill}" opacity="{op:.2f}"/>'


def oblate(x, fill, op, last):
    return (f'<ellipse cx="{x}" cy="{CHORD}" rx="13.5" ry="9.5" fill="{fill}" '
            f'opacity="{op:.2f}" transform="rotate(-22 {x} {CHORD})"/>')


def ringed(x, fill, op, last):
    ring = RING if last else GHOST
    rx, ry = 26, 7.5
    back = f"M {-rx} 0 A {rx} {ry} 0 0 1 {rx} 0"
    front = f"M {rx} 0 A {rx} {ry} 0 0 1 {-rx} 0"
    inner = f"M {-rx + 5} 0 A {rx - 5} {ry - 2} 0 0 1 {rx - 5} 0"
    inner_f = f"M {rx - 5} 0 A {rx - 5} {ry - 2} 0 0 1 {-rx + 5} 0"
    g = (f'<g transform="translate({x} {CHORD}) rotate(-16)" opacity="{op:.2f}">'
         f'<path d="{back}" fill="none" stroke="{ring}" stroke-width="3.2"/>'
         f'<path d="{inner}" fill="none" stroke="{ring}" stroke-width="1.6" opacity="0.8"/>'
         f'<ellipse cx="0" cy="0" rx="11.5" ry="10" fill="{fill}"/>'
         f'<path d="{front}" fill="none" stroke="{ring}" stroke-width="3.2"/>'
         f'<path d="{inner_f}" fill="none" stroke="{ring}" stroke-width="1.6" opacity="0.8"/>'
         '</g>')
    return g


SPOTS = """<g fill="#b45f1e" opacity="0.62">
  <ellipse cx="108" cy="40" rx="9" ry="7.5"/>
  <ellipse cx="143" cy="73" rx="8" ry="7"/>
  <ellipse cx="160" cy="34" rx="4.5" ry="6" transform="rotate(25 160 34)"/>
  <ellipse cx="96" cy="86" rx="5" ry="3.5"/>
</g>
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = {
        "spherical": star() + "\n" + track(sphere),
        "oblate": star() + "\n" + track(oblate),
        "ringed": star() + "\n" + track(ringed, xs=(56, 126)),
        "spots": star(SPOTS) + "\n" + track(sphere, xs=(40, 66, 92, 124)),
    }
    for name, body in pages.items():
        (OUT / f"{name}.svg").write_text(svg(body))
        print("wrote", OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
