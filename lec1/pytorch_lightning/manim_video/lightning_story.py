"""
lightning_story.py  -  3Blue1Brown-style scenes for "From PyTorch to PyTorch Lightning"

Preview one scene :  manim -pql lightning_story.py S1_Model
Render all (1080p):  manim -qh lightning_story.py -a

Colour language, identical in every scene:
    blue   = your model (the science)   -> LightningModule
    green  = your data                  -> LightningDataModule
    gray   = engineering / plumbing     -> Trainer
    yellow = something fires (a hook)   -> callbacks
    purple = logging                    -> TensorBoard
Curves, digits and activations are real: a numpy re-run of the lecture's model
(784-50-10, ReLU, Adam 1e-3, batch 64, 50k/10k split) in assets/lightning_story_data.npz.
"VO:" comments are narration cues.
"""
import re
import sys
from pathlib import Path

import numpy as np
from manim import *

DATA = np.load(Path(__file__).parent / "assets" / "lightning_story_data.npz")

MONO = "Menlo" if sys.platform == "darwin" else "DejaVu Sans Mono"
SERIF = "Palatino" if sys.platform == "darwin" else "DejaVu Serif"

SCI = "#58C4DD"      # model
DAT = "#83C167"      # data
ENG = "#8A8F98"      # engineering -> Trainer
FIRE = "#F4C542"     # hooks
LOG = "#B189F5"      # logging
BAD = "#FC6255"      # bugs
VAL = "#5CD0B3"      # validation arc
TST = "#FF9F5A"      # test arc


# ----------------------------------------------------------------- helpers
def T(s, size=30, color=WHITE, **kw):
    return Text(s, font=SERIF, font_size=size, color=color, **kw)


def M(s, size=20, color=WHITE, **kw):
    return Text(s, font=MONO, font_size=size, color=color, **kw)


class CodeBlock(VGroup):
    """Monospace code; self[i] is line i (a VGroup of glyphs), indentation kept."""

    def __init__(self, src, size=18, color=WHITE, line_spacing=0.75):
        super().__init__()
        self.src_lines = src.strip("\n").split("\n")
        t = Text("\n".join(self.src_lines), font=MONO, font_size=size,
                 color=color, line_spacing=line_spacing)
        k = 0
        for ln in self.src_lines:
            n = len(re.sub(r"\s", "", ln))
            self.add(VGroup(*t.submobjects[k:k + n]))
            k += n

    def part(self, i, sub, occ=0):
        flat, key = re.sub(r"\s", "", self.src_lines[i]), re.sub(r"\s", "", sub)
        start = -1
        for _ in range(occ + 1):
            start = flat.find(key, start + 1)
            if start < 0:
                raise ValueError(f"{sub!r} not found in line {i}")
        return VGroup(*self[i][start:start + len(key)])

    def color_regex(self, pattern, color):
        for i, ln in enumerate(self.src_lines):
            for m in re.finditer(pattern, ln):
                a = len(re.sub(r"\s", "", ln[:m.start()]))
                n = len(re.sub(r"\s", "", m.group()))
                VGroup(*self[i][a:a + n]).set_color(color)
        return self


def hl(mob, color=FIRE, buff=0.07):
    return SurroundingRectangle(mob, color=color, buff=buff, stroke_width=2, corner_radius=0.05)


def chip(label, color=ENG, size=15):
    t = M(label, size, color)
    r = RoundedRectangle(corner_radius=0.08, width=t.width + 0.3, height=0.42,
                         stroke_color=color, stroke_width=1.5)
    t.move_to(r)
    return VGroup(r, t)


def box(label, color=ENG, w=3.0, h=1.0, size=18):
    r = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=color, stroke_width=2)
    t = M(label, size, color).move_to(r)
    return VGroup(r, t)


def digit_img(arr, height=0.5):
    a = np.asarray(arr, dtype=np.uint8)
    rgba = np.dstack([np.full_like(a, 255)] * 3 + [a])
    im = ImageMobject(rgba)
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return im.scale_to_fit_height(height)


def pixel_grid(img, side=2.6):
    s = side / 28
    g = VGroup()
    for r in range(28):
        for c in range(28):
            sq = Square(s, stroke_width=0, fill_color=WHITE, fill_opacity=img[r, c] / 255)
            sq.move_to([(c - 13.5) * s, (13.5 - r) * s, 0])
            g.add(sq)
    return g


def clock_point(center, r, deg):
    """deg measured clockwise from 12 o'clock."""
    a = PI / 2 - deg * DEGREES
    return np.array(center) + r * np.array([np.cos(a), np.sin(a), 0])


def clock_arc(center, r, d0, d1, color, width=8, opacity=0.6):
    return Arc(radius=r, start_angle=PI / 2 - d0 * DEGREES, angle=-(d1 - d0) * DEGREES,
               arc_center=np.array(center), color=color, stroke_width=width, stroke_opacity=opacity)


TRAIN_END, VAL_START, VAL_END = 250, 254, 320


def make_dial(center, r):
    tr = clock_arc(center, r, 0, TRAIN_END, SCI)
    va = clock_arc(center, r, VAL_START, VAL_END, VAL)
    ticks = VGroup()
    for k in range(40):
        d = TRAIN_END * (k + 0.5) / 40
        ticks.add(Line(clock_point(center, r - 0.12, d), clock_point(center, r + 0.12, d),
                       stroke_width=1, color=SCI, stroke_opacity=0.6))
    for k in range(8):
        d = VAL_START + (VAL_END - VAL_START) * (k + 0.5) / 8
        ticks.add(Line(clock_point(center, r - 0.12, d), clock_point(center, r + 0.12, d),
                       stroke_width=1, color=VAL, stroke_opacity=0.6))
    return VGroup(tr, va, ticks)


def swap_text(scene, old, new, run_time=0.25, pos=None):
    new.move_to(old.get_center() if pos is None else pos)
    scene.play(FadeTransform(old, new), run_time=run_time)
    return new


class Dial(VGroup):
    """The loop as a dial. pieces = [("train", ticks), ("val", ticks), ...] laid clockwise from 12 o'clock."""
    COLORS = {"train": SCI, "val": VAL, "test": TST}

    def __init__(self, center, r, pieces, gap=4, tail=40, width=8):
        super().__init__()
        self.c, self.r = np.array(center, dtype=float), r
        per = (360 - tail - gap * (len(pieces) - 1)) / sum(n for _, n in pieces)
        self.ranges, self.arcs, self.ticks = [], VGroup(), VGroup()
        d = 0
        for kind, n in pieces:
            col = self.COLORS[kind]
            self.arcs.add(clock_arc(center, r, d, d + n * per, col, width))
            self.ticks.add(VGroup(*[Line(clock_point(center, r - 0.12, d + per * (k + 0.5)),
                                         clock_point(center, r + 0.12, d + per * (k + 0.5)),
                                         stroke_width=1, color=col, stroke_opacity=0.6) for k in range(n)]))
            self.ranges.append((d, d + n * per, per))
            d += n * per + gap
        self.add(self.arcs, self.ticks)

    def start(self, i):
        return self.ranges[i][0]

    def end(self, i):
        return self.ranges[i][1]

    def tick(self, i, k):
        return self.ranges[i][0] + self.ranges[i][2] * (k + 0.5)


def dial_dot(c, r, theta):
    dot = Dot(clock_point(c, r, 0), radius=0.1, color=FIRE)
    dot.add_updater(lambda m: m.move_to(clock_point(c, r, theta.get_value())))
    return dot


def knob_panel(args, head="pl.Trainer(", size=17):
    h = M(head, size, ENG)
    rows = VGroup(*[M(a, size, GREY_B) for a in args]).arrange(DOWN, aligned_edge=LEFT, buff=0.34)
    rows.next_to(h, DOWN, aligned_edge=LEFT, buff=0.34).shift(0.45 * RIGHT)
    t = M(")", size, ENG).next_to(rows, DOWN, buff=0.34).align_to(h, LEFT)
    return VGroup(h, rows, t)


def pick(scene, panel, i, box=None):
    """Light row i of a knob panel, dim the rest, move the highlight box."""
    rows = panel[1]
    nb = hl(rows[i])
    anims = [r.animate.set_color(WHITE if j == i else GREY_D) for j, r in enumerate(rows)]
    anims.append(ReplacementTransform(box, nb) if box is not None else Create(nb))
    scene.play(*anims, run_time=0.8)
    return nb


def caption(s, x=2.6, y=-3.05, size=24, color=GREY_A, max_w=8.6):
    c = T(s, size, color).move_to([x, y, 0])
    return c.scale_to_fit_width(min(c.width, max_w))


# ======================================================== 0. the tangled script
RAW_LEFT = '''class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
device = torch.device("cuda" if cuda else "cpu")
batch_size, num_epochs, lr = 64, 3, 0.001
entire = datasets.MNIST("dataset/", train=True)
train_ds, val_ds = random_split(entire, [50000, 10000])
test_ds = datasets.MNIST("dataset/", train=False)
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
test_loader = DataLoader(test_ds, batch_size)
model = NN(784, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)'''
LEFT_CAT = "SSSSSSSS" + "E" + "C" + "DDDDDD" + "SSS"

RAW_RIGHT = '''for epoch in range(num_epochs):
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        data = data.reshape(data.shape[0], -1)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def check_accuracy(loader, model):
    num_correct = num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.reshape(x.shape[0], -1)
            _, preds = model(x).max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
    model.train()
    return num_correct / num_samples
print(check_accuracy(train_loader, model))
print(check_accuracy(val_loader, model))
print(check_accuracy(test_loader, model))'''
RIGHT_CAT = "EEEE" + "SSS" + "EEE" + "EEEEEE" + "SS" + "EEEE" + "EEE"


class S0_Tangled(Scene):
    def construct(self):
        left, right = CodeBlock(RAW_LEFT, 14), CodeBlock(RAW_RIGHT, 14)
        right.next_to(left, RIGHT, buff=0.7, aligned_edge=UP)
        code = VGroup(left, right)
        code.scale_to_fit_width(min(code.width, 13.4)).move_to(ORIGIN).to_edge(UP, buff=0.35)
        lines, cats = [*left, *right], LEFT_CAT + RIGHT_CAT

        # VO: "This is the whole script from lesson one. It trains an MNIST classifier, and it works."
        self.play(LaggedStart(*[FadeIn(l, shift=0.1 * RIGHT) for l in lines],
                              lag_ratio=0.04, run_time=3))
        self.wait(1.5)

        # VO: "But read it again: it's really three different things tangled together."
        groups = {c: VGroup(*[l for l, k in zip(lines, cats) if k == c]) for c in "SDEC"}
        mixed = left.part(16, ".to(device)")
        for c, color, label in [("S", SCI, "your model"), ("D", DAT, "your data"),
                                ("E", ENG, "plumbing")]:
            cap = T(label, 32, color).to_edge(DOWN, buff=0.3)
            extra = [mixed.animate.set_color(ENG)] if c == "E" else []
            self.play(groups[c].animate.set_color(color), *extra, FadeIn(cap, shift=0.2 * UP), run_time=1.2)
            self.wait(1.2)
            self.play(FadeOut(cap), run_time=0.4)

        # VO: "Sort the lines by job, and the lines themselves become a bar chart."
        cols = {"S": (-5.0, SCI, "LightningModule"), "D": (-1.7, DAT, "LightningDataModule"),
                "E": (1.8, ENG, "Trainer"), "C": (5.2, WHITE, "config.py")}
        moves, base = [], -2.45
        for c, (x, color, name) in cols.items():
            for i, l in enumerate(groups[c]):
                l.generate_target()
                l.target.scale(0.48).move_to([x, base + 0.15 * i, 0])
                moves.append(MoveToTarget(l))
        self.play(LaggedStart(*moves, lag_ratio=0.02, run_time=3.5))
        labels, counts = VGroup(), VGroup()
        for c, (x, color, name) in cols.items():
            n = len(groups[c])
            labels.add(M(name, 16, color).move_to([x, -3.05, 0]))
            counts.add(T(f"{n} line{'s' if n > 1 else ''}", 22, color)
                       .move_to([x, base + 0.15 * n + 0.2, 0]))
        self.play(FadeIn(counts, shift=0.2 * UP), run_time=1)
        self.wait(1)
        # VO: "Half of it is plumbing, and it's the same plumbing in every project you'll ever write.
        #      Lightning's whole idea: give each of these three its own home."
        self.play(Write(labels), run_time=2)
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


# ================================================================ 1. the model
NN_SRC = '''class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x'''

LOOP_BODY = '''for data, targets in train_loader:
    data = data.to(device)
    targets = targets.to(device)
    data = data.reshape(data.shape[0], -1)
    scores = model(data)
    loss = criterion(scores, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()'''

TRAIN_STEP = '''def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.reshape(x.size(0), -1)
    scores = self.forward(x)
    loss = self.loss_fn(scores, y)
    return loss'''


class S1_Model(Scene):
    def construct(self):
        k = 7
        img, hidden, probs = DATA["digits"][k], DATA["hidden"][k], DATA["probs"][k]

        # ---- A. what the network is (homage to "But what is a neural network?")
        grid = pixel_grid(img, 2.6).move_to([-4.4, 0.3, 0])
        frame = SurroundingRectangle(grid, color=GREY_D, buff=0.02, stroke_width=1)
        shape = M("x: [28, 28]", 18, GREY_B).next_to(frame, DOWN, buff=0.25)
        code = CodeBlock(NN_SRC, 16).to_edge(RIGHT, buff=0.45).shift(0.9 * UP)
        for i in range(9):
            code[i].set_color(SCI)
        reshape = M("x = x.reshape(x.size(0), -1)", 16, SCI).next_to(code, DOWN, buff=0.6, aligned_edge=LEFT)

        # VO: "Start with the part that is genuinely yours: the model."
        self.play(FadeIn(grid), Create(frame), FadeIn(shape), FadeIn(code, lag_ratio=0.05), run_time=2)
        self.wait()

        # VO: "A 28x28 image becomes a column of 784 numbers."
        self.play(FadeIn(reshape, shift=0.2 * UP))
        rb = hl(reshape)
        self.play(Create(rb), run_time=0.6)
        top, h, colx = 3.3, 6.4 / 784, -5.9
        rows = VGroup(*[VGroup(*grid[r * 28:(r + 1) * 28]) for r in range(28)])
        anims = []
        for r, row in enumerate(rows):
            for c, sq in enumerate(row):
                sq.generate_target()
                sq.target.stretch_to_fit_width(0.24).stretch_to_fit_height(h)
                sq.target.move_to([colx, top - (r * 28 + c + 0.5) * h, 0])
            anims.append(AnimationGroup(*[MoveToTarget(sq) for sq in row]))
        self.play(FadeOut(frame), FadeOut(shape), FadeOut(rb),
                  LaggedStart(*anims, lag_ratio=0.06, run_time=3.5))
        n784 = M("784", 18, GREY_B).move_to([colx, -3.55, 0])
        self.play(FadeIn(n784))

        # layers 784 -> 50 -> 10
        hx, ox = -3.6, -1.4
        hid = VGroup(*[Circle(0.05, stroke_color=GREY_B, stroke_width=1).move_to([hx, 3.2 - i * 0.128, 0])
                       for i in range(50)])
        out = VGroup(*[Circle(0.17, stroke_color=GREY_B, stroke_width=1.5).move_to([ox, 2.5 - i * 0.55, 0])
                       for i in range(10)])
        digits = VGroup(*[T(str(i), 22, GREY_B).next_to(out[i], RIGHT, buff=0.2) for i in range(10)])
        n50 = M("50", 18, GREY_B).move_to([hx, -3.55, 0])
        n10 = M("10", 18, GREY_B).move_to([ox, -3.55, 0])
        idx = np.linspace(20, 763, 24).astype(int)
        e1 = VGroup(*[Line([colx + 0.12, top - (i + 0.5) * h, 0], c.get_center(),
                           stroke_width=0.6, stroke_opacity=0.12, color=SCI) for i in idx for c in hid])
        e2 = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=0.6, stroke_opacity=0.12, color=SCI)
                      for a in hid for b in out])
        self.play(Create(e1, lag_ratio=0.001), Create(e2, lag_ratio=0.002), FadeIn(hid), FadeIn(out),
                  FadeIn(digits), FadeIn(n50), FadeIn(n10), run_time=2.5)

        # VO: "fc1 is every one of those wires: 784 times 50 weights."
        b1 = hl(code[3])
        self.play(Create(b1), e1.animate.set_stroke(opacity=0.45), run_time=1)
        self.play(e1.animate.set_stroke(opacity=0.12), run_time=0.6)
        # VO: "ReLU: each hidden neuron lights up, or stays dark."
        b2 = hl(code[6])
        hmax = hidden.max()
        self.play(ReplacementTransform(b1, b2),
                  *[c.animate.set_fill(WHITE, float(v / hmax)) for c, v in zip(hid, hidden)], run_time=1.2)
        # VO: "fc2 turns 50 numbers into 10 scores, one per digit."
        b3 = hl(code[7])
        self.play(ReplacementTransform(b2, b3), e2.animate.set_stroke(opacity=0.4), run_time=1)
        best = int(np.argmax(probs))
        self.play(e2.animate.set_stroke(opacity=0.12),
                  *[c.animate.set_fill(WHITE, float(p)) for c, p in zip(out, probs)],
                  digits[best].animate.set_color(FIRE).scale(1.4), run_time=1.2)
        acc = T(f"{float(DATA['test_acc']) * 100:.1f}% of test digits right after 3 epochs", 22, GREY_B)
        acc.next_to(reshape, DOWN, buff=0.5, aligned_edge=LEFT)
        self.play(FadeIn(acc), FadeOut(b3))
        self.wait(2)

        # ---- B. nn.Module -> pl.LightningModule
        net = VGroup(rows, e1, e2, hid, out, digits, n784, n50, n10)
        self.play(FadeOut(net), FadeOut(reshape), FadeOut(acc), code.animate.to_edge(LEFT, buff=0.6))
        head = CodeBlock("class NN(pl.LightningModule):", 16)[0].set_color(SCI)
        head.move_to(code[0], aligned_edge=LEFT)
        # VO: "Change one word."
        self.play(TransformMatchingShapes(code[0], head), run_time=1.5)
        outer = RoundedRectangle(corner_radius=0.3, width=5.2, height=3.4, stroke_color=GREY_B, stroke_width=2)
        outer.move_to([3.6, 0.3, 0])
        inner = RoundedRectangle(corner_radius=0.2, width=4.0, height=1.5, stroke_color=SCI, stroke_width=2)
        inner.move_to(outer).shift(0.45 * DOWN)
        lo = M("nn.Module", 18, GREY_B).next_to(outer.get_corner(UL), DR, buff=0.25)
        li = M("pl.LightningModule", 18, SCI).move_to(inner)
        note = T("still an nn.Module, plus a contract", 24, GREY_A).next_to(outer, DOWN, buff=0.35)
        # VO: "A LightningModule is still an nn.Module, so everything you know still works.
        #      It just adds a contract: a few methods with agreed names."
        self.play(Create(outer), FadeIn(lo))
        self.play(Create(inner), FadeIn(li))
        self.play(FadeIn(note, shift=0.2 * UP))
        self.wait(2)
        self.play(FadeOut(VGroup(outer, inner, lo, li, note, head, *code[1:])))

        # ---- C. the loop body becomes training_step
        loop = CodeBlock(LOOP_BODY, 17).to_edge(LEFT, buff=0.5).shift(0.6 * UP)
        for i in range(9):
            loop[i].set_color(SCI if i in (3, 4, 5) else ENG)
        ts = CodeBlock(TRAIN_STEP, 17).to_edge(RIGHT, buff=0.5).shift(0.6 * UP)
        ts.set_color(SCI)
        trainer = box("Trainer", ENG, 2.6, 0.9).to_corner(DR, buff=0.5)
        cap = T("one turn of the plain PyTorch loop", 24, GREY_B).next_to(loop, UP, buff=0.4, aligned_edge=LEFT)
        self.play(FadeIn(loop, lag_ratio=0.05), FadeIn(cap), run_time=1.5)
        self.wait()
        # VO: "The gray lines are the same in every project, so the Trainer takes them."
        self.play(FadeIn(trainer))
        self.play(LaggedStart(*[loop[i].animate.scale(0.3).move_to(trainer).set_opacity(0)
                                for i in (1, 2, 6, 7, 8)], lag_ratio=0.15, run_time=2.2),
                  Indicate(trainer, color=ENG, scale_factor=1.05, run_time=2.4))
        # VO: "The loop header becomes a signature: the Trainer owns the loop and hands you one batch."
        self.play(FadeOut(cap), FadeTransform(loop[0], VGroup(ts[0], ts[1])), run_time=1.5)
        self.play(*[TransformMatchingShapes(loop[i], ts[j]) for i, j in ((3, 2), (4, 3), (5, 4))], run_time=1.8)
        self.play(Write(ts[5]))
        arr = Arrow(ts[5].get_right() + 0.1 * RIGHT, trainer.get_top(), color=ENG, stroke_width=3, buff=0.1)
        lab = M("backward(), step()", 15, ENG).next_to(trainer, UP, buff=0.15).shift(1.4 * LEFT)
        # VO: "You return the loss. The Trainer calls backward and step for you."
        self.play(GrowArrow(arr), FadeIn(lab))
        self.wait(2)
        self.play(FadeOut(VGroup(ts, trainer, arr, lab)))

        # ---- D. the contract: you fill slots, the Trainer decides when they run
        names = ["__init__", "forward", "training_step", "validation_step", "test_step",
                 "predict_step", "configure_optimizers"]
        slots = VGroup(*[box(n, SCI, 3.6, 0.44, 16) for n in names]).arrange(DOWN, buff=0.12)
        card_head = M("class NN(pl.LightningModule)", 16, SCI)
        card = VGroup(card_head, slots).arrange(DOWN, buff=0.25).move_to([-0.6, 0, 0])
        frame = SurroundingRectangle(card, color=SCI, buff=0.2, corner_radius=0.15, stroke_width=1.5)
        tr = box("Trainer", ENG, 2.0, 4.6).move_to([5.2, slots.get_center()[1], 0])
        self.play(FadeIn(card, lag_ratio=0.1), Create(frame), FadeIn(tr), run_time=1.5)
        calls = [(6, "once, at start"), (2, "x 782 / epoch"), (3, "x 157 / epoch"),
                 (4, "trainer.test()"), (5, "trainer.predict()")]
        # VO: "Notice you never call these yourself. The Trainer calls them, at the right moment."
        for i, txt in calls:
            s = slots[i]
            a = Arrow([tr.get_left()[0], s.get_center()[1], 0], s.get_right(), buff=0.08,
                      color=ENG, stroke_width=2.5, max_tip_length_to_length_ratio=0.08)
            l = M(txt, 14, GREY_B).next_to(a, UP, buff=0.05)
            self.play(GrowArrow(a), FadeIn(l), run_time=0.5)
            self.play(Indicate(s, color=FIRE, scale_factor=1.06), run_time=0.7)
        self.wait()
        # VO: "And because train, validation and test share one body, you write the science once."
        common = box("_common_step", SCI, 2.5, 0.9, 16).move_to([-5.3, slots[3].get_center()[1], 0])
        sub = M("reshape, forward, loss", 12, SCI).next_to(common, DOWN, buff=0.12)
        self.play(FadeIn(common), FadeIn(sub))
        self.play(*[GrowArrow(Arrow(slots[i].get_left(), common.get_right(), buff=0.08, color=SCI,
                                    stroke_width=2.5, max_tip_length_to_length_ratio=0.1))
                    for i in (2, 3, 4)], run_time=1.2)
        tag = T("you fill the slots, the Trainer decides when they run", 24, GREY_A).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(tag, shift=0.2 * UP))
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)))


# ===================================== 1b. inside the module: __init__, summary, optimizers
INIT_SRC = '''class NN(pl.LightningModule):
    def __init__(self, input_size, learning_rate, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros(1, input_size)'''

SUMMARY = '''  | Name    | Type             | Params | In sizes | Out sizes
0 | fc1     | Linear           | 39.2 K | [1, 784] | [1, 50]
1 | fc2     | Linear           | 510    | [1, 50]  | [1, 10]
2 | loss_fn | CrossEntropyLoss | 0      | ?        | ?
39.8 K    Total params
0.159     Total estimated model params size (MB)'''

OPT_V1 = '''def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)'''

OPT_V2 = '''def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    return {"optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}'''


class S1b_ModelSlots(Scene):
    def construct(self):
        # ---- A. __init__ + save_hyperparameters -> hparams.yaml and the checkpoint
        head = T("__init__: build the parts, and remember how", 26, GREY_A).to_corner(UL, buff=0.4)
        code = CodeBlock(INIT_SRC, 14, SCI).next_to(head, DOWN, buff=0.4, aligned_edge=LEFT)
        # VO: "Back inside the LightningModule. Every slot has a few things worth knowing, starting with __init__."
        self.play(FadeIn(head), FadeIn(code, lag_ratio=0.05), run_time=1.5)
        b = hl(code[3])
        hp = RoundedRectangle(corner_radius=0.15, width=3.5, height=1.7, stroke_color=SCI, stroke_width=2)
        hp.move_to([4.6, 2.2, 0])
        hpl = M("self.hparams", 15, SCI).next_to(hp, UP, buff=0.1)
        toks = VGroup(M("input_size: 784", 14), M("learning_rate: 0.001", 14), M("num_classes: 10", 14))
        toks.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(hp)
        srcs = [code.part(1, "input_size"), code.part(1, "learning_rate"), code.part(1, "num_classes")]
        # VO: "save_hyperparameters() grabs the arguments of __init__ and keeps them in self.hparams."
        self.play(Create(b))
        self.play(Create(hp), FadeIn(hpl),
                  *[TransformFromCopy(s, t) for s, t in zip(srcs, toks)], run_time=1.6)
        self.wait()
        yaml = box("hparams.yaml", LOG, 2.4, 0.55, 14).move_to([2.1, 0.45, 0])
        ylab = M("tb_logs/.../version_0/", 12, GREY_B).next_to(yaml, DOWN, buff=0.1)
        layers = VGroup(*[box(s, c, 3.0, 0.4, 13) for s, c in
                          [("state_dict (weights)", SCI), ("optimizer_states", ENG),
                           ("epoch, global_step", ENG), ("hyper_parameters", FIRE)]]).arrange(DOWN, buff=0.06)
        ck_head = M("epoch=2-step=2346.ckpt", 13, GREY_A)
        ckpt = VGroup(ck_head, layers).arrange(DOWN, buff=0.12).move_to([5.1, -0.55, 0])
        ckl = M(".../version_0/checkpoints/", 12, GREY_B).next_to(ckpt, DOWN, buff=0.1)
        a1 = Arrow(hp.get_bottom(), yaml.get_top(), buff=0.1, color=GREY_B, stroke_width=2.5)
        # VO: "They land in two places: next to your TensorBoard logs, and inside every checkpoint.
        #      The Trainer saves a checkpoint for you at the end of training: 3 epochs, 2,346 steps."
        self.play(GrowArrow(a1), FadeIn(yaml), FadeIn(ylab))
        self.play(FadeIn(ckpt, lag_ratio=0.15), FadeIn(ckl), run_time=1.5)
        self.play(TransformFromCopy(toks, layers[3]), run_time=1)
        self.wait()
        load = CodeBlock('model = NN.load_from_checkpoint(ckpt_path)', 15, SCI).move_to([-3.2, -2.5, 0])
        load.color_regex(r"load_from_checkpoint", FIRE)
        restored = box("NN, same weights", SCI, 2.9, 0.65, 15).move_to([1.6, -2.5, 0])
        # VO: "So reloading needs no arguments at all. The checkpoint already knows them."
        self.play(FadeIn(load), FadeOut(b))
        self.play(TransformFromCopy(layers, restored), run_time=1.5)
        noargs = M("no arguments needed", 14, GREY_B).next_to(restored, DOWN, buff=0.12)
        self.play(FadeIn(noargs))
        self.wait(2)
        self.play(FadeOut(VGroup(hp, hpl, toks, yaml, ylab, a1, ckpt, ckl, load, restored, noargs)))

        # ---- B. the model summary, and where the parameters live
        b2 = hl(code[7])
        # VO: "One more line: example_input_array. Lightning runs it through the model once, to print shapes."
        self.play(Create(b2))
        self.wait()
        self.play(FadeOut(code), FadeOut(b2), FadeOut(head))
        head2 = T("the model summary, printed when fit() starts", 26, GREY_A).to_corner(UL, buff=0.4)
        table = CodeBlock(SUMMARY, 15, GREY_B).next_to(head2, DOWN, buff=0.4).set_x(0)
        for i in (1, 2):
            table.part(i, "39.2 K" if i == 1 else "510").set_color(SCI)
        shapes = VGroup(table.part(0, "In sizes | Out sizes"), table.part(1, "[1, 784] | [1, 50]"),
                        table.part(2, "[1, 50]  | [1, 10]"))
        self.play(FadeIn(head2), FadeIn(table, lag_ratio=0.1), run_time=1.5)
        sb = hl(shapes)
        self.play(Create(sb), shapes.animate.set_color(FIRE))
        self.wait()
        s = 0.0085
        w1 = Rectangle(width=784 * s, height=50 * s, stroke_color=SCI, stroke_width=1.5,
                       fill_color=SCI, fill_opacity=0.35).move_to([0.4, -1.0, 0])
        w2 = Rectangle(width=50 * s, height=10 * s, stroke_color=SCI, stroke_width=1.5,
                       fill_color=SCI, fill_opacity=0.35).next_to(w1, DOWN, buff=0.7, aligned_edge=LEFT)
        l1 = M("fc1.weight [50, 784]", 14, SCI).next_to(w1, UP, buff=0.12, aligned_edge=LEFT)
        l2 = M("fc2.weight [10, 50]", 14, SCI).next_to(w2, RIGHT, buff=0.25)
        # VO: "Draw each weight matrix to scale, and you see where your 39,760 parameters live."
        self.play(FadeOut(sb), TransformFromCopy(table.part(1, "39.2 K"), w1), FadeIn(l1), run_time=1.5)
        self.play(TransformFromCopy(table.part(2, "510"), w2), FadeIn(l2), run_time=1.2)
        pct = T("98.7% of the parameters are in fc1", 26, GREY_A).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(pct, shift=0.2 * UP))
        self.wait(2.5)
        self.play(FadeOut(VGroup(head2, table, w1, w2, l1, l2, pct)))

        # ---- C. configure_optimizers: an optimizer, optionally with a scheduler
        head3 = T("configure_optimizers: what to step, and how fast", 26, GREY_A).to_corner(UL, buff=0.4)
        v1 = CodeBlock(OPT_V1, 15, SCI).next_to(head3, DOWN, buff=0.4, aligned_edge=LEFT)
        # VO: "The simplest contract: return an optimizer. Note self.hparams, thanks to save_hyperparameters."
        self.play(FadeIn(head3), FadeIn(v1))
        self.play(Indicate(v1.part(1, "self.hparams.learning_rate"), color=FIRE))
        v2 = CodeBlock(OPT_V2, 15, SCI).move_to(v1, aligned_edge=UL)
        v2.color_regex(r"StepLR\(optimizer, step_size=5, gamma=0.5\)", FIRE).color_regex(r'"lr_scheduler"', FIRE)
        # VO: "Or return a dictionary with a learning-rate scheduler, and the Trainer steps it for you."
        self.play(FadeTransform(v1, v2), run_time=1.2)
        ax = Axes(x_range=[0, 20, 5], y_range=[0, 0.0011, 0.00025], x_length=9, y_length=3.0, tips=False,
                  axis_config={"color": GREY_C, "stroke_width": 1.5}).move_to([-1.0, -1.75, 0])
        yl = M("lr", 15, SCI).next_to(ax.y_axis, UP, buff=0.1)
        xl = M("epoch", 13, GREY_B).next_to(ax.x_axis, DOWN, buff=0.15).align_to(ax.x_axis, RIGHT)
        xt = VGroup(*[M(str(e), 12, GREY_B).next_to(ax.c2p(e, 0), DOWN, buff=0.12) for e in (0, 5, 10, 15)])
        self.play(Create(ax), FadeIn(yl), FadeIn(xl), FadeIn(xt))
        lrs = [0.001 * 0.5 ** (e // 5) for e in range(20)]
        for k in range(4):
            seg = Line(ax.c2p(5 * k, lrs[5 * k]), ax.c2p(5 * k + 5, lrs[5 * k]), color=SCI, stroke_width=4)
            val = M(["1e-3", "5e-4", "2.5e-4", "1.25e-4"][k], 13, SCI).next_to(seg, UP, buff=0.08)
            anims = [Create(seg), FadeIn(val)]
            if k:
                drop = DashedLine(ax.c2p(5 * k, lrs[5 * k - 1]), ax.c2p(5 * k, lrs[5 * k]), color=FIRE)
                anims = [Create(drop), Flash(ax.c2p(5 * k, lrs[5 * k]), color=FIRE, line_length=0.15)] + anims
            self.play(*anims, run_time=0.9)
        side = VGroup(T("every 5 epochs:", 22, GREY_A), M("lr x 0.5", 18, FIRE)).arrange(DOWN, buff=0.15)
        side.move_to([5.4, -1.2, 0])
        self.play(FadeIn(side))
        plateau = VGroup(T("ReduceLROnPlateau instead?", 20, GREY_B), M('add "monitor": "val_loss"', 14, FIRE))
        plateau.arrange(DOWN, buff=0.12).move_to([5.1, -2.5, 0])
        plateau.scale_to_fit_width(min(plateau.width, 3.6))
        # VO: "Prefer to cut the rate only when validation stops improving? ReduceLROnPlateau, plus a monitor key."
        self.play(FadeIn(plateau))
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)))


# ======================================== 1c. the module at work: modes, self.log, predict
NOT_WRITTEN = '''model.train()
model.eval()
with torch.no_grad():
batch = batch.to(device)
optimizer.zero_grad()
loss.backward()
optimizer.step()'''

PREDICT_SRC = '''def predict_step(self, batch, batch_idx):
    x, y = batch
    x = x.reshape(x.size(0), -1)
    scores = self.forward(x)
    preds = torch.argmax(scores, dim=1)
    return preds'''


class S1c_ModelAtWork(Scene):
    def construct(self):
        # ---- A. what the Trainer switches for you, and what the module can see
        head = T("lines you never write again", 26, GREY_A).to_corner(UL, buff=0.4)
        nw = CodeBlock(NOT_WRITTEN, 16, ENG).next_to(head, DOWN, buff=0.5, aligned_edge=LEFT)
        strikes = VGroup(*[Line(l.get_left() + 0.08 * LEFT, l.get_right() + 0.08 * RIGHT, color=BAD,
                                stroke_width=2.5) for l in nw])
        # VO: "Seven lines from plain PyTorch you will never type again. The Trainer does them, at the right moment."
        self.play(FadeIn(head), FadeIn(nw, lag_ratio=0.1), run_time=1.5)
        self.play(LaggedStart(*[Create(s) for s in strikes], lag_ratio=0.15), run_time=1.5)
        C, R = np.array([0.6, 0.0, 0]), 1.95
        dial = Dial(C, R, [("train", 40), ("val", 8)])
        theta, ep = ValueTracker(0), ValueTracker(0)
        dot = dial_dot(C, R, theta)
        t_end, v0, v1 = dial.end(0), dial.start(1), dial.end(1)

        def in_val():
            return theta.get_value() >= v0 - 0.5

        lx, vx, ys = 3.0, 5.35, [1.6, 0.9, 0.2, -0.5]
        labels = VGroup(*[M(s, 15, GREY_B).move_to([lx, y, 0], aligned_edge=LEFT) for s, y in
                          zip(["mode", "gradients", "self.current_epoch", "self.global_step"], ys)])
        mode = always_redraw(lambda: M("model.eval()" if in_val() else "model.train()", 15,
                                       VAL if in_val() else SCI).move_to([vx, ys[0], 0], aligned_edge=LEFT))
        grad = always_redraw(lambda: M("off" if in_val() else "on", 15,
                                       BAD if in_val() else DAT).move_to([vx, ys[1], 0], aligned_edge=LEFT))
        cur = always_redraw(lambda: M(str(int(ep.get_value())), 15, WHITE).move_to([vx, ys[2], 0], aligned_edge=LEFT))
        gstep = always_redraw(lambda: M(str(int(ep.get_value()) * 782 + int(782 * min(theta.get_value(), t_end) / t_end)),
                                        15, WHITE).move_to([vx, ys[3], 0], aligned_edge=LEFT))
        self.play(Create(dial), FadeIn(dot), FadeIn(labels), FadeIn(mode), FadeIn(grad), FadeIn(cur),
                  FadeIn(gstep), run_time=1.5)
        note = caption("validation, test and predict run in eval mode, without gradients", x=1.2, y=-2.6,
                       size=22, max_w=9.5)
        dev = caption("and every batch arrives already on self.device", x=1.2, y=-3.15, size=22, max_w=9.5)
        # VO: "Watch the panel. Training: train mode, gradients on. Validation: eval mode, gradients off.
        #      Inside any method you can read self.current_epoch and self.global_step."
        for e, (rt_t, rt_v) in enumerate([(4, 1.6), (1.6, 0.8)]):
            ep.set_value(e)
            theta.set_value(0)
            self.play(theta.animate.set_value(t_end), run_time=rt_t, rate_func=linear)
            self.play(theta.animate.set_value(v1), run_time=rt_v, rate_func=linear)
            if e == 0:
                self.play(FadeIn(note), FadeIn(dev))
            self.play(theta.animate.set_value(360), run_time=0.4, rate_func=linear)
        self.wait(1.5)
        for m in (mode, grad, cur, gstep):
            m.clear_updaters()
        dot.clear_updaters()
        self.play(FadeOut(Group(*self.mobjects)))

        # ---- B. self.log: one call, three destinations
        head2 = T("self.log: one value, three destinations", 26, GREY_A).to_corner(UL, buff=0.4)
        call = CodeBlock('self.log("val_loss", loss, prog_bar=True, logger=True)', 17)
        call.next_to(head2, DOWN, buff=0.35, aligned_edge=LEFT)
        call.color_regex(r"self\.log", LOG).color_regex(r"prog_bar=True", FIRE).color_regex(r"logger=True", LOG)
        v = float(DATA["epoch_val"][2])
        src = box(f"val_loss = {v:.3f}", VAL, 3.0, 0.7, 16).move_to([-4.9, -0.6, 0])
        bar = VGroup(RoundedRectangle(corner_radius=0.1, width=6.1, height=0.62, stroke_color=GREY_B, stroke_width=1.5),
                     M(f"Epoch 2: 100%|██████████| 782/782 [val_loss={v:.3f}]", 13, GREY_A))
        bar[1].move_to(bar[0])
        bar.move_to([3.65, 1.1, 0])
        tb = box("logger: TensorBoard", LOG, 3.2, 0.62, 15).move_to([2.2, -0.6, 0])
        cm = box('trainer.callback_metrics["val_loss"]', FIRE, 5.0, 0.62, 14).move_to([3.0, -2.3, 0])
        users = VGroup(chip("EarlyStopping(monitor=...)", FIRE, 13), chip("ModelCheckpoint(monitor=...)", FIRE, 13))
        users.arrange(RIGHT, buff=0.25).next_to(cm, DOWN, buff=0.5)
        sx = src.get_right()[0]
        paths = [VMobject().set_points_as_corners([src.get_right(), [sx + 0.6, src.get_y(), 0],
                                                   [sx + 0.6, d.get_y(), 0], d.get_left()])
                 for d in (bar, tb, cm)]
        cols = [FIRE, LOG, GREY_B]
        for p, c in zip(paths, cols):
            p.set_stroke(c, 2)
        tags = VGroup(M("prog_bar=True", 13, FIRE).next_to(paths[0], UP, buff=0.05).shift(0.6 * RIGHT),
                      M("logger=True", 13, LOG).next_to(tb, UP, buff=0.12),
                      M("always", 13, GREY_B).next_to(cm, UP, buff=0.12))
        # VO: "self.log sends one value to three places. prog_bar puts it on the progress bar.
        #      logger sends it to TensorBoard. And it always lands in callback_metrics,
        #      which is exactly what EarlyStopping and ModelCheckpoint read when you write monitor='val_loss'."
        self.play(FadeIn(head2), FadeIn(call))
        self.play(FadeIn(src))
        self.play(*[Create(p) for p in paths], FadeIn(bar), FadeIn(tb), FadeIn(cm), FadeIn(tags), run_time=1.5)
        pulses = [Dot(color=c, radius=0.08).move_to(src.get_right()) for c in cols]
        self.play(*[MoveAlongPath(pd, p) for pd, p in zip(pulses, paths)], run_time=1.5)
        self.play(*[FadeOut(pd) for pd in pulses], Indicate(bar), Indicate(tb), Indicate(cm))
        a = Arrow(cm.get_bottom(), users.get_top(), buff=0.08, color=FIRE, stroke_width=2.5)
        self.play(GrowArrow(a), FadeIn(users))
        self.wait(2)
        off = CodeBlock('self.log("val_loss", loss, prog_bar=False, logger=True)', 17).move_to(call, aligned_edge=LEFT)
        off.color_regex(r"self\.log", LOG).color_regex(r"logger=True", LOG)
        # VO: "Turn prog_bar off and the progress bar goes quiet. Nothing else changes."
        self.play(FadeTransform(call, off), paths[0].animate.set_stroke(opacity=0.15),
                  bar[1].animate.set_opacity(0.2), tags[0].animate.set_opacity(0.2))
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)))

        # ---- C. predict_step: real predictions from the 3-epoch model
        head3 = T("predict_step: the model in use", 26, GREY_A).to_corner(UL, buff=0.4)
        pc = CodeBlock(PREDICT_SRC, 15, SCI).next_to(head3, DOWN, buff=0.4, aligned_edge=LEFT)
        call2 = CodeBlock("preds = trainer.predict(model, dataloaders=test_loader)", 15, ENG)
        call2.next_to(pc, DOWN, buff=0.45, aligned_edge=LEFT).color_regex("model", SCI).color_regex("test_loader", DAT)
        # VO: "Last slot: predict_step. trainer.predict runs it over a dataloader, in eval mode, and hands back predictions."
        self.play(FadeIn(head3), FadeIn(pc, lag_ratio=0.1), FadeIn(call2), run_time=1.5)
        idx = list(range(10, 18))
        probs, labels = DATA["probs"], DATA["labels"]
        ims = Group(*[digit_img(DATA["digits"][i], 0.62) for i in idx]).arrange(RIGHT, buff=0.12)
        ims.move_to([3.6, 1.7, 0])
        pbox = box("predict_step", SCI, 3.4, 0.7, 16).move_to([3.6, 0.2, 0])
        self.play(FadeIn(ims, lag_ratio=0.1), FadeIn(pbox))
        self.play(*[im.copy().animate.scale(0.3).move_to(pbox).set_opacity(0) for im in ims], run_time=1.2)
        outs = VGroup()
        for k, i in enumerate(idx):
            p, t = int(np.argmax(probs[i])), int(labels[i])
            outs.add(T(str(p), 30, DAT if p == t else BAD).move_to([ims[k].get_x(), -1.3, 0]))
        self.play(LaggedStart(*[FadeIn(o, shift=0.4 * DOWN) for o in outs], lag_ratio=0.1), run_time=1.5)
        wrong = [k for k, i in enumerate(idx) if int(np.argmax(probs[i])) != int(labels[i])]
        for k in wrong:
            true = M(f"label {int(labels[idx[k]])}", 13, BAD).next_to(outs[k], DOWN, buff=0.15)
            self.play(Circumscribe(Group(ims[k], outs[k]), color=BAD), FadeIn(true))
        res = T("157 test batches -> a list of 157 tensors", 22, GREY_A).move_to([3.6, -2.7, 0])
        self.play(FadeIn(res))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


# ============================================================== 2. the trainer
TRAINER_LOOP = '''model.to(device)
for epoch in range(max_epochs):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = to_device(batch, device)
        loss = model.training_step(batch, batch_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            model.validation_step(batch, batch_idx)'''

TRAINER_SRC = '''trainer = pl.Trainer(
    accelerator="cpu",
    devices=1,
    min_epochs=1,
    max_epochs=3,
    precision=16,
)
trainer.fit(model, train_loader, val_loader)
trainer.validate(model, val_loader)
trainer.test(model, test_loader)'''


class S2_Trainer(Scene):
    def construct(self):
        code = CodeBlock(TRAINER_LOOP, 16).to_edge(LEFT, buff=0.4).shift(0.3 * UP)
        code.set_color(ENG)
        code.part(5, "model.training_step(batch, batch_idx)").set_color(SCI)
        code.part(12, "model.validation_step(batch, batch_idx)").set_color(SCI)
        C, R = np.array([3.9, 0.3, 0]), 2.1
        dial = make_dial(C, R)
        theta = ValueTracker(0)
        dot = Dot(clock_point(C, R, 0), radius=0.1, color=FIRE)
        dot.add_updater(lambda m: m.move_to(clock_point(C, R, theta.get_value())))
        ep = T("epoch 1 / 3", 26).move_to(C + 0.25 * UP)
        phase = M("", 15, GREY_B).move_to(C + 0.3 * DOWN)
        cap = T("the loop, written by you, every time", 24, GREY_B).to_edge(UP, buff=0.3).shift(3 * LEFT)

        # VO: "Here's the loop every PyTorch project writes, now calling your training_step."
        self.play(FadeIn(cap), FadeIn(code, lag_ratio=0.05), run_time=1.5)
        # VO: "Draw it as what it is: a loop. One turn is one epoch."
        self.play(Create(dial), FadeIn(dot), FadeIn(ep), run_time=2)
        self.add(phase)
        box_tr = hl(VGroup(*code[3:9]), SCI)
        phase = swap_text(self, phase, M("training_step x 782", 15, SCI), pos=C + 0.3 * DOWN)
        self.play(Create(box_tr), theta.animate.set_value(TRAIN_END), run_time=4, rate_func=linear)
        box_va = hl(VGroup(*code[9:13]), VAL)
        phase = swap_text(self, phase, M("validation_step x 157", 15, VAL))
        self.play(ReplacementTransform(box_tr, box_va),
                  theta.animate.set_value(VAL_END), run_time=1.6, rate_func=linear)
        self.play(theta.animate.set_value(360), FadeOut(box_va), run_time=0.6, rate_func=linear)
        for e in (2, 3):
            theta.set_value(0)
            ep = swap_text(self, ep, T(f"epoch {e} / 3", 26), 0.2)
            self.play(theta.animate.set_value(360), run_time=1.4, rate_func=linear)
        theta.set_value(0)
        self.wait()

        # VO: "None of this is your research. So Lightning seals it inside a machine."
        tcode = CodeBlock(TRAINER_SRC, 16).to_edge(LEFT, buff=0.4).shift(0.3 * UP)
        tcode.set_color(ENG)
        for i, sub in ((7, "model"), (8, "model"), (9, "model")):
            tcode.part(i, sub).set_color(SCI)
        for i, sub in ((7, "train_loader"), (7, "val_loader"), (8, "val_loader"), (9, "test_loader")):
            tcode.part(i, sub).set_color(DAT)
        machine = RoundedRectangle(corner_radius=0.3, width=5.4, height=5.6, stroke_color=ENG, stroke_width=2.5)
        machine.move_to(C)
        mlabel = M("pl.Trainer", 18, ENG).next_to(machine.get_corner(UL), DR, buff=0.2)
        self.play(FadeOut(cap), FadeOut(phase),
                  LaggedStart(*[l.animate.scale(0.2).move_to(tcode[7]).set_opacity(0) for l in code],
                              lag_ratio=0.05, run_time=2),
                  Create(machine), FadeIn(mlabel))
        self.play(FadeIn(tcode, lag_ratio=0.05), run_time=1.5)
        # VO: "The loop didn't disappear. You just don't write it any more."
        self.wait(1.5)

        # knob: max_epochs
        b = hl(tcode[4])
        self.play(Create(b))
        for e in (1, 2, 3):
            ep = swap_text(self, ep, T(f"epoch {e} / 3", 26), 0.15)
            self.play(theta.animate.set_value(360), run_time=0.8, rate_func=linear)
            theta.set_value(0)
        self.play(Flash(dot, color=FIRE), run_time=0.6)

        # knob: precision
        b2 = hl(tcode[5])
        self.play(ReplacementTransform(b, b2))
        cells32 = VGroup(*[Square(0.12, stroke_width=1, stroke_color=ENG, fill_color=SCI, fill_opacity=0.6)
                           for _ in range(32)]).arrange(RIGHT, buff=0.02)
        cells32.move_to([C[0] + 0.7, -3.25, 0])
        lab32 = M("float32", 15, GREY_B).next_to(cells32, LEFT, buff=0.2)
        self.play(FadeIn(cells32, lag_ratio=0.05), FadeIn(lab32))
        lab16 = M("16-bit math (mixed)", 15, GREY_B).next_to(cells32[:16], LEFT, buff=0.2)
        # VO: "precision=16: do the arithmetic in half the bits. Faster, lighter, same code."
        self.play(FadeOut(cells32[16:], shift=0.3 * RIGHT), FadeTransform(lab32, lab16), run_time=1.2)
        self.wait()
        self.play(FadeOut(cells32[:16]), FadeOut(lab16))

        # fit / validate / test
        tr_arc, va_arc = dial[0], dial[1]
        b3 = hl(tcode[7])
        self.play(ReplacementTransform(b2, b3))
        self.play(theta.animate.set_value(360), run_time=1.4, rate_func=linear)
        theta.set_value(0)
        b4 = hl(tcode[8])
        phase = M("validation_step", 15, VAL).move_to(C + 0.3 * DOWN)
        self.play(ReplacementTransform(b3, b4), tr_arc.animate.set_stroke(opacity=0.12),
                  ep.animate.set_opacity(0), FadeIn(phase))
        theta.set_value(VAL_START)
        self.play(theta.animate.set_value(VAL_END), run_time=1.2, rate_func=linear)
        b5 = hl(tcode[9])
        tphase = M("test_step", 15, TST).move_to(phase)
        self.play(ReplacementTransform(b4, b5), va_arc.animate.set_stroke(color=TST),
                  FadeTransform(phase, tphase))
        theta.set_value(VAL_START)
        self.play(theta.animate.set_value(VAL_END), run_time=1.2, rate_func=linear)
        tag = T("same machine, three ways to run it", 24, GREY_A).to_edge(DOWN, buff=0.25).shift(3.3 * LEFT)
        self.play(FadeIn(tag))
        self.wait(2.5)
        dot.clear_updaters()
        self.play(FadeOut(Group(*self.mobjects)))


# ===================================================== 2b. Trainer arguments for debugging
class S2b_TrainerDebug(Scene):
    def construct(self):
        head = T("Trainer arguments for debugging", 26, GREY_A).to_corner(UL, buff=0.4)
        panel = knob_panel(["num_sanity_val_steps=2", "fast_dev_run=True",
                            "limit_train_batches=0.25", "overfit_batches=1"])
        panel.next_to(head, DOWN, buff=0.6, aligned_edge=LEFT)
        C, R = np.array([3.4, 0.35, 0]), 2.05
        dial = Dial(C, R, [("train", 40), ("val", 8)])
        per = dial.ranges[0][2]
        theta = ValueTracker(0)
        dot = dial_dot(C, R, theta)
        # VO: "The Trainer has dozens of arguments. A few of them exist purely to help you find bugs fast."
        self.play(FadeIn(head), FadeIn(panel, lag_ratio=0.1), Create(dial), FadeIn(dot), run_time=1.5)

        # num_sanity_val_steps (default 2)
        bx = pick(self, panel, 0)
        cap = caption("runs 2 validation batches before training,\nso a bug in validation_step shows up in seconds",
                      x=C[0], size=22, max_w=6.8)
        lab = M("sanity check", 15, FIRE).move_to(C)
        sticks = dial.ticks[1][:2]
        # VO: "num_sanity_val_steps: before the first epoch, run two validation batches. It's on by default."
        self.play(sticks.animate.set_stroke(FIRE, width=3, opacity=1), FadeIn(cap), FadeIn(lab))
        theta.set_value(dial.start(1))
        self.play(theta.animate.set_value(dial.tick(1, 1) + per / 2), run_time=1.4, rate_func=linear)
        self.wait(0.5)
        theta.set_value(0)
        self.play(sticks.animate.set_stroke(VAL, width=1, opacity=0.6), FadeOut(lab), run_time=0.3)
        self.play(theta.animate.set_value(360), run_time=2.2, rate_func=linear)
        theta.set_value(0)

        # fast_dev_run
        bx = pick(self, panel, 1, bx)
        keep = VGroup(dial.ticks[0][0], dial.ticks[1][0])
        cap2 = caption("1 training batch, 1 validation batch, then stop.\nNo logs, no checkpoints: a smoke test",
                       x=C[0], size=22, max_w=6.8)
        # VO: "fast_dev_run: one batch of each and stop. Run it every time you change the code."
        self.play(FadeOut(cap), FadeIn(cap2), dial.arcs.animate.set_stroke(opacity=0.1),
                  dial.ticks.animate.set_stroke(opacity=0.1))
        self.play(keep.animate.set_stroke(FIRE, width=3, opacity=1))
        self.play(theta.animate.set_value(dial.tick(0, 0)), run_time=0.5)
        self.play(Flash(dot, color=FIRE, line_length=0.15), run_time=0.5)
        theta.set_value(dial.tick(1, 0))
        self.play(Flash(dot, color=FIRE, line_length=0.15), run_time=0.5)
        done = M("done", 18, FIRE).move_to(C)
        self.play(FadeIn(done))
        self.wait(1.5)
        theta.set_value(0)
        self.play(FadeOut(done), dial.arcs.animate.set_stroke(opacity=0.6),
                  dial.ticks[0].animate.set_stroke(SCI, width=1, opacity=0.6),
                  dial.ticks[1].animate.set_stroke(VAL, width=1, opacity=0.6))

        # limit_train_batches
        bx = pick(self, panel, 2, bx)
        bright = clock_arc(C, R, 0, 10 * per, SCI, opacity=1)
        cap3 = caption("each epoch uses only the first 25% of the training batches;\n"
                       "limit_val_batches and limit_test_batches work the same way", x=C[0], size=22, max_w=6.8)
        cap3b = VGroup()
        # VO: "limit_train_batches: a quarter of the data per epoch. Quick experiments, same code path."
        self.play(FadeOut(cap2), FadeIn(cap3), FadeIn(cap3b), dial.arcs[0].animate.set_stroke(opacity=0.1),
                  dial.ticks[0][10:].animate.set_stroke(opacity=0.1), Create(bright))
        for _ in range(2):
            theta.set_value(0)
            self.play(theta.animate.set_value(10 * per), run_time=1.2, rate_func=linear)
            theta.set_value(dial.start(1))
            self.play(theta.animate.set_value(dial.end(1)), run_time=0.7, rate_func=linear)
            self.play(theta.animate.set_value(360), run_time=0.3, rate_func=linear)
        self.wait()

        # overfit_batches
        bx = pick(self, panel, 3, bx)
        dot.clear_updaters()
        over = DATA["overfit_losses"]
        ax = Axes(x_range=[0, 200, 50], y_range=[0, 2.4, 0.6], x_length=5.8, y_length=3.6, tips=False,
                  axis_config={"color": GREY_C, "stroke_width": 1.5}).move_to(C)
        yl = M("train_loss", 14, SCI).next_to(ax.y_axis, UP, buff=0.1)
        xt = VGroup(*[M(str(e), 12, GREY_B).next_to(ax.c2p(e, 0), DOWN, buff=0.12) for e in (0, 50, 100, 150, 200)])
        xl = M("epoch, always the same 64 images", 13, GREY_B).next_to(xt, DOWN, buff=0.12).align_to(ax.x_axis, RIGHT)
        curve = VMobject(stroke_color=SCI, stroke_width=3).set_points_as_corners(
            [ax.c2p(i, float(v)) for i, v in enumerate(over)])
        cap4 = caption("can the model memorize one batch?\nif the loss doesn't fall to ~0, something is broken",
                       x=C[0], size=22, max_w=6.8)
        # VO: "overfit_batches: train on one batch, over and over. A working model memorizes it.
        #      This is the real curve: from 2.29 to under 0.01."
        self.play(FadeOut(VGroup(dial, dot, bright, cap3, cap3b)), FadeIn(cap4))
        self.play(Create(ax), FadeIn(yl), FadeIn(xt), FadeIn(xl))
        self.play(Create(curve), run_time=3, rate_func=linear)
        endv = M(f"{float(over[-1]):.3f}", 15, FIRE).next_to(ax.c2p(200, float(over[-1])), UP, buff=0.2)
        self.play(FadeIn(endv), Flash(ax.c2p(200, float(over[-1])), color=FIRE))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


# ================================================ 2c. Trainer arguments that reshape the loop
class S2c_TrainerLoop(Scene):
    def construct(self):
        head = T("Trainer arguments that reshape the loop", 26, GREY_A).to_corner(UL, buff=0.4)
        panel = knob_panel(["val_check_interval=0.25", "check_val_every_n_epoch=2",
                            "accumulate_grad_batches=4", "gradient_clip_val=0.5"])
        panel.next_to(head, DOWN, buff=0.6, aligned_edge=LEFT)
        C, R = np.array([3.4, 0.35, 0]), 2.05
        default = [("train", 40), ("val", 8)]
        dial = Dial(C, R, default)
        per = dial.ranges[0][2]
        theta = ValueTracker(0)
        dot = dial_dot(C, R, theta)
        self.play(FadeIn(head), FadeIn(panel, lag_ratio=0.1), Create(dial), FadeIn(dot), run_time=1.5)

        # val_check_interval
        bx = pick(self, panel, 0)
        quarter = Dial(C, R, [("train", 10), ("val", 3)] * 4)
        cap = caption("validate 4 times per epoch\n(an int means: every N training batches)", x=C[0], size=22,
                      max_w=6.8)
        # VO: "val_check_interval: don't wait for the end of the epoch to validate. Check four times per lap."
        self.play(FadeTransform(dial, quarter), FadeIn(cap), run_time=1.2)
        dial = quarter
        for i in range(8):
            self.play(theta.animate(rate_func=linear).set_value(dial.end(i)),
                      *([Indicate(dial.arcs[i], color=FIRE, scale_factor=1.0)] if i % 2 else []),
                      run_time=0.8 if i % 2 == 0 else 0.45)
        self.play(theta.animate.set_value(360), run_time=0.3, rate_func=linear)
        theta.set_value(0)
        back = Dial(C, R, default)
        self.play(FadeTransform(dial, back), FadeOut(cap))
        dial = back

        # check_val_every_n_epoch
        bx = pick(self, panel, 1, bx)
        cap2 = caption("validate only every 2nd epoch:\nuseful when validation is slow", x=C[0], size=22, max_w=6.8)
        epl = T("epoch 0", 24).move_to(C)
        # VO: "check_val_every_n_epoch: the opposite, validate less often."
        self.play(FadeIn(cap2), FadeIn(epl), dial.arcs[1].animate.set_stroke(opacity=0.1),
                  dial.ticks[1].animate.set_stroke(opacity=0.1))
        self.play(theta.animate.set_value(360), run_time=2.0, rate_func=linear)
        theta.set_value(0)
        self.play(epl.animate.become(T("epoch 1", 24).move_to(C)), dial.arcs[1].animate.set_stroke(opacity=0.9),
                  dial.ticks[1].animate.set_stroke(opacity=0.6), run_time=0.4)
        self.play(theta.animate.set_value(dial.end(0)), run_time=1.4, rate_func=linear)
        self.play(theta.animate(rate_func=linear).set_value(dial.end(1)),
                  Indicate(dial.arcs[1], color=FIRE, scale_factor=1.0), run_time=0.8)
        self.play(theta.animate.set_value(360), run_time=0.3, rate_func=linear)
        theta.set_value(0)
        self.play(FadeOut(epl), FadeOut(cap2), dial.arcs[1].animate.set_stroke(opacity=0.6))

        # accumulate_grad_batches
        bx = pick(self, panel, 2, bx)
        bounds = VGroup(*[dial.ticks[0][k] for k in range(3, 40, 4)])
        lamp = Circle(0.13, stroke_color=FIRE, stroke_width=2).move_to([C[0] + 1.3, C[1] + 2.45, 0])
        lampl = M("optimizer.step()", 14, FIRE).next_to(lamp, RIGHT, buff=0.15)
        cap3 = caption("add up gradients from 4 batches, then step once:\neffective batch 64 x 4 = 256",
                       x=C[0], size=22, max_w=6.8)
        cap3b = caption("global_step counts optimizer steps:\n196 per epoch instead of 782", x=C[0], size=22,
                        max_w=6.8)
        # VO: "accumulate_grad_batches: four batches of gradients add up, then one step.
        #      It's how you get a bigger batch than your memory allows."
        self.play(bounds.animate.set_stroke(FIRE, width=3, opacity=1), FadeIn(lamp), FadeIn(lampl), FadeIn(cap3))
        for g in range(4):
            self.play(theta.animate.set_value(dial.tick(0, 4 * g + 3) + per / 2), run_time=1.0, rate_func=linear)
            self.play(lamp.animate.set_fill(FIRE, 1), run_time=0.12)
            self.play(lamp.animate.set_fill(FIRE, 0), run_time=0.25)
        self.play(FadeOut(cap3), FadeIn(cap3b))
        self.play(theta.animate.set_value(360), run_time=1.5, rate_func=linear)
        theta.set_value(0)
        self.wait()

        # gradient_clip_val (+ gradient_clip_algorithm)
        bx = pick(self, panel, 3, bx)
        dot.clear_updaters()
        self.play(FadeOut(VGroup(dial, dot, lamp, lampl, cap3, cap3b)))
        plane = NumberPlane(x_range=[-1.2, 1.2, 0.5], y_range=[-1.2, 1.2, 0.5], x_length=4.8, y_length=4.8,
                            background_line_style={"stroke_color": GREY_D, "stroke_width": 1, "stroke_opacity": 0.6},
                            axis_config={"stroke_color": GREY_C, "stroke_width": 1.5}).move_to(C)
        O = plane.c2p(0, 0)
        u = plane.x_axis.unit_size
        ring = Circle(radius=0.5 * u, color=FIRE, stroke_width=2.5).move_to(O)
        g = np.array([0.9, 0.55])
        arrow = Arrow(O, plane.c2p(*g), buff=0, color=WHITE, stroke_width=4)
        gl = M(f"gradient, length {np.linalg.norm(g):.2f}", 14, WHITE).next_to(arrow.get_end(), UP, buff=0.1)
        cap4 = caption("norm (the default): longer than 0.5? shrink it.\nThe direction stays the same.", x=C[0],
                       size=22, max_w=6.8)
        # VO: "gradient_clip_val: one bad batch can produce a huge gradient. Clipping caps its length."
        self.play(Create(plane), Create(ring), run_time=1.2)
        self.play(GrowArrow(arrow), FadeIn(gl))
        self.wait(0.5)
        ghost = DashedLine(O, plane.c2p(*g), color=GREY_B, stroke_width=2)
        gn = g * 0.5 / np.linalg.norm(g)
        self.add(ghost)
        self.play(arrow.animate.put_start_and_end_on(O, plane.c2p(*gn)).set_color(DAT), FadeIn(cap4),
                  gl.animate.become(M("length 0.50", 14, DAT).next_to(plane.c2p(*gn), UP, buff=0.12)), run_time=1.5)
        self.wait(1.5)
        alg = M('gradient_clip_algorithm="value"', 15, FIRE).next_to(panel, DOWN, buff=0.35, aligned_edge=LEFT)
        sq = Square(side_length=u, color=FIRE, stroke_width=2.5).move_to(O)
        gv = np.clip(g, -0.5, 0.5)
        cap5 = caption("value: clip each component to [-0.5, 0.5].\nSimpler, but the direction changes.",
                       x=C[0], size=22, max_w=6.8)
        # VO: "Or clip by value: each component separately. Notice the arrow now points somewhere else."
        self.play(FadeIn(alg), ReplacementTransform(ring, sq), FadeOut(cap4), FadeIn(cap5),
                  arrow.animate.put_start_and_end_on(O, plane.c2p(*g)).set_color(WHITE), FadeOut(gl))
        self.play(arrow.animate.put_start_and_end_on(O, plane.c2p(*gv)).set_color(TST), run_time=1.5)
        ang = M(f"{np.degrees(np.arctan2(*g[::-1])):.0f}° -> {np.degrees(np.arctan2(*gv[::-1])):.0f}°",
                15, TST).next_to(plane.c2p(*gv), UR, buff=0.1)
        self.play(FadeIn(ang))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


# ========================================================= 2d. Trainer arguments for hardware
SCALE_SRC = '''trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0, 1],
    strategy="ddp",
    precision="16-mixed",
)'''


class S2d_TrainerScale(Scene):
    def construct(self):
        head = T("Trainer arguments for hardware", 26, GREY_A).to_corner(UL, buff=0.4)
        code = CodeBlock(SCALE_SRC, 17, ENG).next_to(head, DOWN, buff=0.6, aligned_edge=LEFT)
        accs = VGroup(*[chip(a, ENG, 14) for a in ('"cpu"', '"gpu"', '"mps"', '"auto"')]).arrange(RIGHT, buff=0.15)
        accs.next_to(code, DOWN, buff=0.5, aligned_edge=LEFT)
        accn = M("mps = Apple silicon GPUs", 12, GREY_B).next_to(accs, DOWN, buff=0.12, aligned_edge=LEFT)
        # VO: "Same model, same code. Moving to a GPU, or to several, is a Trainer argument."
        self.play(FadeIn(head), FadeIn(code, lag_ratio=0.1), run_time=1.2)
        b = hl(code[1])
        self.play(Create(b), FadeIn(accs, lag_ratio=0.1), FadeIn(accn))
        self.play(accs[1].animate.set_color(FIRE))

        # one device -> two devices with DistributedSampler + all-reduce
        C1, R1 = np.array([3.6, 0.3, 0]), 1.6
        d1 = Dial(C1, R1, [("train", 40), ("val", 8)])
        l1 = M("1 GPU: 782 batches / epoch", 14, GREY_B).next_to(d1, DOWN, buff=0.25)
        data = Rectangle(width=5.6, height=0.32, stroke_width=0, fill_color=DAT, fill_opacity=0.55).move_to([3.6, 2.75, 0])
        dl = M("50,000 training images", 13, DAT).next_to(data, UP, buff=0.08)
        self.play(Create(d1), FadeIn(l1), FadeIn(data), FadeIn(dl))
        b2 = hl(code[2])
        Ca, Cb, r = np.array([1.75, 0.2, 0]), np.array([5.45, 0.2, 0]), 1.2
        da, db = Dial(Ca, r, [("train", 20), ("val", 4)]), Dial(Cb, r, [("train", 20), ("val", 4)])
        la = M("GPU 0: 391 batches", 13, GREY_B).next_to(da, DOWN, buff=0.2)
        lb = M("GPU 1: 391 batches", 13, GREY_B).next_to(db, DOWN, buff=0.2)
        ha = Rectangle(width=2.75, height=0.32, stroke_width=0, fill_color=DAT, fill_opacity=0.55).move_to([1.75, 2.75, 0])
        hb = ha.copy().move_to([5.45, 2.75, 0])
        hla = M("25,000", 13, DAT).next_to(ha, UP, buff=0.08)
        hlb = M("25,000", 13, DAT).next_to(hb, UP, buff=0.08)
        # VO: "devices=[0, 1]: two GPUs. Each one gets its own copy of the model and half of the data."
        self.play(ReplacementTransform(b, b2), FadeTransform(d1, VGroup(da, db)),
                  FadeTransform(l1, VGroup(la, lb)), ReplacementTransform(data, VGroup(ha, hb)),
                  FadeTransform(dl, VGroup(hla, hlb)), run_time=1.6)
        ma = box("NN", SCI, 1.4, 0.5, 15).move_to([1.75, -2.0, 0])
        mb = box("NN", SCI, 1.4, 0.5, 15).move_to([5.45, -2.0, 0])
        sync = DoubleArrow(ma.get_right(), mb.get_left(), buff=0.12, color=FIRE, stroke_width=3,
                           max_tip_length_to_length_ratio=0.06)
        syl = M("all-reduce: average the gradients", 14, FIRE).next_to(sync, DOWN, buff=0.4)
        b3 = hl(code[3])
        self.play(ReplacementTransform(b2, b3), FadeIn(ma), FadeIn(mb))
        ta, tb = ValueTracker(0), ValueTracker(0)
        dota, dotb = dial_dot(Ca, r, ta), dial_dot(Cb, r, tb)
        self.add(dota, dotb)
        # VO: "strategy='ddp': after every backward pass the GPUs average their gradients,
        #      so both copies take exactly the same step and stay identical."
        step = da.end(0) / 4
        for k in range(4):
            self.play(ta.animate.set_value(step * (k + 1)), tb.animate.set_value(step * (k + 1)),
                      run_time=0.8, rate_func=linear)
            if k == 0:
                self.play(GrowFromCenter(sync), FadeIn(syl), run_time=0.6)
            else:
                self.play(Indicate(sync, color=FIRE, scale_factor=1.05), run_time=0.4)
        self.play(ta.animate.set_value(360), tb.animate.set_value(360), run_time=0.8, rate_func=linear)
        sd = CodeBlock('self.log("val_loss", loss, sync_dist=True)', 15, WHITE)
        sd.color_regex(r"sync_dist=True", FIRE).next_to(accn, DOWN, buff=0.5, aligned_edge=LEFT)
        va, vb = M("a", 18, VAL).next_to(ma, LEFT, buff=0.2), M("b", 18, VAL).next_to(mb, RIGHT, buff=0.2)
        avg = M("logged: (a + b) / 2", 16, VAL).next_to(syl, DOWN, buff=0.3)
        # VO: "And when you log on several GPUs, sync_dist=True averages the value across them."
        self.play(FadeIn(sd), FadeIn(va), FadeIn(vb))
        self.play(TransformFromCopy(VGroup(va, vb), avg), run_time=1.2)
        self.wait(1.5)
        dota.clear_updaters()
        dotb.clear_updaters()

        # DeepSpeed ZeRO-2, as in folder 10
        ds = M("strategy=DeepSpeedStrategy(),", 17, FIRE).move_to(code[3], aligned_edge=LEFT)
        self.play(FadeOut(VGroup(da, db, la, lb, ha, hb, hla, hlb, dota, dotb, ma, mb, sync, syl, va, vb, avg)),
                  FadeTransform(code[3], ds), FadeOut(b3))
        u, x0 = 1.0, 1.4
        rows, lbls = VGroup(), VGroup()
        for gi, y in enumerate((1.3, -0.3)):
            parts = VGroup(Rectangle(width=u, height=0.55, fill_color=SCI, fill_opacity=0.5, stroke_width=1),
                           Rectangle(width=u, height=0.55, fill_color=ENG, fill_opacity=0.5, stroke_width=1),
                           Rectangle(width=2 * u, height=0.55, fill_color=FIRE, fill_opacity=0.5, stroke_width=1))
            parts.arrange(RIGHT, buff=0).move_to([x0, y, 0], aligned_edge=LEFT)
            rows.add(parts)
            lbls.add(M(f"GPU {gi}", 14, GREY_B).next_to(parts, LEFT, buff=0.2))
        names = VGroup(*[VGroup(Square(0.18, fill_color=c, fill_opacity=0.6, stroke_width=0), M(t, 13, c)).arrange(RIGHT, buff=0.1)
                         for t, c in (("weights", SCI), ("gradients", ENG), ("Adam state", FIRE))])
        names.arrange(RIGHT, buff=0.4).move_to([3.4, 2.4, 0])
        ddp = M('strategy="ddp": everything copied', 14, GREY_B).next_to(rows[1], DOWN, buff=0.35, aligned_edge=LEFT)
        # VO: "Your folder-10 code uses DeepSpeed. With plain DDP, every GPU holds everything:
        #      weights, gradients, and Adam's state, which is twice the size of the weights."
        self.play(FadeIn(rows), FadeIn(lbls), FadeIn(names), FadeIn(ddp))
        self.wait()
        new_rows = VGroup()
        for gi, parts in enumerate(rows):
            np_ = VGroup(parts[0].copy(),
                         Rectangle(width=0.5 * u, height=0.55, fill_color=ENG, fill_opacity=0.5, stroke_width=1),
                         Rectangle(width=u, height=0.55, fill_color=FIRE, fill_opacity=0.5, stroke_width=1))
            np_.arrange(RIGHT, buff=0).move_to(parts, aligned_edge=LEFT)
            new_rows.add(np_)
        halves = VGroup(*[M(f"shard {gi}", 11, WHITE).move_to(new_rows[gi][2]) for gi in range(2)])
        zero = T("ZeRO stage 2 (the default): gradients and\noptimizer state are split between GPUs, not copied",
                 20, GREY_A)
        zero.scale_to_fit_width(min(zero.width, 5.6)).move_to([3.9, -1.35, 0])
        # VO: "DeepSpeed's ZeRO stage 2 splits the gradients and the optimizer state between the GPUs.
        #      Each one keeps only its shard, so a bigger model fits."
        self.play(ReplacementTransform(rows, new_rows), FadeOut(ddp), FadeIn(zero), run_time=1.6)
        self.play(FadeIn(halves))
        self.wait(1)
        b4 = hl(code[4])
        prec = VGroup(*[chip(p, ENG, 13) for p in ('"32-true"', '"16-mixed"', '"bf16-mixed"')]).arrange(RIGHT, buff=0.15)
        prec.move_to([3.6, -2.75, 0])
        precl = M("default 32-true; 16-mixed / bf16-mixed do the math in 16 bits", 13, GREY_B)
        precl.next_to(prec, DOWN, buff=0.15)
        # VO: "And precision, which you saw already: the same string-valued knob."
        self.play(Create(b4), FadeIn(prec), FadeIn(precl))
        self.play(prec[1].animate.set_color(FIRE))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


# ========================================================== 3. the data module
class S3_DataModule(Scene):
    def construct(self):
        # VO: "The green lines, the data, get a home of their own too: the LightningDataModule."
        # ---- A. the factory: prepare_data -> setup -> dataloaders
        frame = RoundedRectangle(corner_radius=0.3, width=13.4, height=6.4, stroke_color=DAT, stroke_width=2)
        frame.move_to(0.35 * DOWN)
        head = M("class MnistDataModule(pl.LightningDataModule)", 17, DAT).next_to(frame.get_top(), DOWN, buff=0.25)
        self.play(Create(frame), FadeIn(head))
        xs = [-4.6, -0.6, 4.0]
        titles = VGroup(*[M(s, 16, DAT).move_to([x, 1.85, 0]) for s, x in
                          zip(["prepare_data()", "setup(stage)", "*_dataloader()"], xs)])
        divs = VGroup(*[DashedLine([x, 2.1, 0], [x, -3.3, 0], color=GREY_D, stroke_width=1) for x in (-2.6, 1.6)])
        self.play(FadeIn(titles), Create(divs))

        # station 1
        disk = box("dataset/", GREY_B, 2.0, 0.6, 15).move_to([xs[0], -0.2, 0])
        src = M("MNIST", 16, GREY_B).move_to([xs[0], 1.1, 0])
        dl = Arrow(src.get_bottom(), disk.get_top(), buff=0.1, color=GREY_B, stroke_width=3)
        n1 = T("download once, on one process", 18, GREY_B).move_to([xs[0], -1.4, 0])
        # VO: "prepare_data: things you do once, like downloading."
        self.play(FadeIn(src), GrowArrow(dl), FadeIn(disk), FadeIn(n1), run_time=1.5)
        self.play(Indicate(disk, color=FIRE), run_time=0.8)

        # station 2
        full = Rectangle(width=3.0, height=0.36, stroke_width=0, fill_color=DAT, fill_opacity=0.55)
        full.move_to([xs[1], 0.9, 0])
        lf = M("60,000", 14, DAT).next_to(full, UP, buff=0.1)
        self.play(FadeIn(full), FadeIn(lf))
        tr = Rectangle(width=2.5, height=0.36, stroke_width=0, fill_color=DAT, fill_opacity=0.55)
        va = Rectangle(width=0.5, height=0.36, stroke_width=0, fill_color=VAL, fill_opacity=0.7)
        tr.move_to(full, aligned_edge=LEFT).shift(0.25 * DOWN + 0.1 * LEFT)
        va.move_to(full, aligned_edge=RIGHT).shift(0.25 * DOWN + 0.2 * RIGHT)
        ltr = M("train 50,000", 13, DAT).next_to(tr, DOWN, buff=0.12)
        lva = M("val 10,000", 13, VAL).next_to(va, DOWN, buff=0.12).shift(0.2 * RIGHT)
        te = Rectangle(width=0.5, height=0.36, stroke_width=0, fill_color=TST, fill_opacity=0.6)
        te.move_to([xs[1] + 1.25, -1.2, 0])
        lte = M("test 10,000", 13, TST).next_to(te, DOWN, buff=0.12)
        n2 = T("runs on every process", 18, GREY_B).move_to([xs[1], -2.4, 0])
        # VO: "setup: build the datasets and split them. random_split, 50,000 and 10,000."
        self.play(ReplacementTransform(full, VGroup(tr, va)), FadeOut(lf), run_time=1.4)
        self.play(FadeIn(ltr), FadeIn(lva), FadeIn(te), FadeIn(lte), FadeIn(n2))

        # station 3: three belts of real digits
        belts, rows_img, ys = VGroup(), [], [0.85, -0.55, -1.95]
        names = [("train_dataloader()", DAT, "shuffle=True"), ("val_dataloader()", VAL, "shuffle=False"),
                 ("test_dataloader()", TST, "shuffle=False")]
        for j, (y, (nm, col, sh)) in enumerate(zip(ys, names)):
            belts.add(Line([2.0, y - 0.32, 0], [6.4, y - 0.32, 0], color=col, stroke_width=2))
            belts.add(M(nm, 13, col).move_to([3.4, y + 0.42, 0]))
            belts.add(M(sh, 12, GREY_B).move_to([5.6, y + 0.42, 0]))
            ims = Group(*[digit_img(DATA["digits"][10 + 6 * j + i], 0.5) for i in range(6)])
            ims.arrange(RIGHT, buff=0.18).move_to([4.2, y, 0])
            rows_img.append(ims)
        # VO: "And the dataloaders: conveyor belts that hand out batches of 64."
        self.play(FadeIn(belts), *[FadeIn(r, lag_ratio=0.1) for r in rows_img], run_time=1.5)
        for ep in range(2):
            self.play(*[r.animate.shift(0.3 * RIGHT) for r in rows_img], run_time=0.5)
            self.play(*[r.animate.shift(0.3 * LEFT) for r in rows_img], run_time=0.3)
            perm = [3, 0, 5, 1, 4, 2] if ep == 0 else [2, 4, 0, 5, 3, 1]
            pos = [m.get_center() for m in rows_img[0]]
            # VO: "Train reshuffles every epoch. Validation and test never do."
            self.play(*[rows_img[0][i].animate.move_to(pos[p]) for i, p in enumerate(perm)],
                      Indicate(belts[2], color=FIRE), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)))

        # ---- B. plug both cartridges into the same Trainer
        fit = M("trainer.fit(model, dm)", 22).to_edge(UP, buff=0.6)
        fit[12:17].set_color(SCI)
        fit[18:20].set_color(DAT)
        machine = RoundedRectangle(corner_radius=0.3, width=5.6, height=4.6, stroke_color=ENG, stroke_width=2.5)
        machine.move_to([2.6, -0.6, 0])
        mlabel = M("pl.Trainer", 18, ENG).next_to(machine.get_corner(UL), DR, buff=0.2)
        mcard = box("model: NN", SCI, 3.0, 0.8, 17).move_to([-4.6, 0.6, 0])
        dcard = box("dm: MnistDataModule", DAT, 3.4, 0.8, 17).move_to([-4.6, -1.0, 0])
        self.play(Write(fit), Create(machine), FadeIn(mlabel), FadeIn(mcard), FadeIn(dcard))
        self.play(mcard.animate.scale(0.75).move_to(machine.get_corner(UL) + np.array([1.4, -1.1, 0])),
                  dcard.animate.scale(0.75).move_to(machine.get_corner(UL) + np.array([3.9, -1.1, 0])),
                  run_time=1.5)
        seq = ["dm.prepare_data()", 'dm.setup("fit")', "dm.train_dataloader()",
               "dm.val_dataloader()", "model.training_step(...)"]
        calls = VGroup(*[M(s, 15, GREY_B) for s in seq]).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        calls.move_to(machine).shift(0.55 * DOWN)
        # VO: "One line. The Trainer knows the order: download, set up, then ask for batches."
        for c in calls:
            self.play(FadeIn(c, shift=0.1 * RIGHT), run_time=0.35)
            self.play(c.animate.set_color(FIRE), run_time=0.25)
            self.play(c.animate.set_color(GREY_B), run_time=0.2)
        self.wait()
        test = M("trainer.test(model, dm)", 22).move_to(fit)
        test[13:18].set_color(SCI)
        test[19:21].set_color(DAT)
        seq2 = VGroup(M('dm.setup("test")', 15, GREY_B), M("dm.test_dataloader()", 15, GREY_B),
                      M("model.test_step(...)", 15, GREY_B)).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        seq2.move_to(calls, aligned_edge=UL)
        # VO: "Same object for testing. The split is defined once, so it can't drift."
        self.play(FadeTransform(fit, test), FadeOut(calls), FadeIn(seq2, lag_ratio=0.3), run_time=1.5)
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


# ============================================================= 4. TensorBoard
class S4_TensorBoard(Scene):
    def construct(self):
        steps = DATA["step_losses"]
        n = len(steps)
        ax = Axes(x_range=[0, n, n / 3], y_range=[0, 2.4, 0.6], x_length=7.4, y_length=4.2, tips=False,
                  axis_config={"color": GREY_C, "stroke_width": 1.5})
        ax.to_edge(LEFT, buff=0.6).shift(0.5 * DOWN)
        ylab = M("train_loss", 15, LOG).next_to(ax.y_axis, UP, buff=0.15)
        xl = VGroup(*[M(f"epoch {e}", 13, GREY_B).next_to(ax.c2p(n / 6 + e * n / 3, 0), DOWN, buff=0.2)
                      for e in range(3)])
        RX = 1.6
        code1 = CodeBlock('self.log("train_loss", loss)', 16).move_to([RX, 2.6, 0], aligned_edge=LEFT)
        code1.color_regex(r"self\.log", LOG).color_regex(r'"train_loss"', FIRE)
        # VO: "self.log writes a number down every time training_step runs."
        self.play(Create(ax), FadeIn(ylab), FadeIn(xl), FadeIn(code1))
        pts = [ax.c2p(i, float(v)) for i, v in enumerate(steps)]
        segs = [VMobject(stroke_color=LOG, stroke_width=1.4).set_points_as_corners(pts[e * n // 3:(e + 1) * n // 3 + 1])
                for e in range(3)]
        self.play(*[Create(s) for s in segs], run_time=4, rate_func=linear)
        note = T("2,346 training steps", 22, GREY_B).next_to(code1, DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(FadeIn(note))
        self.wait()
        # VO: "But the Trainer doesn't write every one. By default it writes one step in fifty."
        knob = CodeBlock("pl.Trainer(log_every_n_steps=50)", 15, ENG).next_to(note, DOWN, buff=0.45, aligned_edge=LEFT)
        knob.color_regex(r"log_every_n_steps=50", FIRE)
        dflt = M("# the default", 13, GREY_C).next_to(knob, DOWN, buff=0.12, aligned_edge=LEFT)
        sdots = VGroup(*[Dot(pts[i], radius=0.045, color=LOG) for i in range(49, n, 50)])
        self.play(FadeIn(knob), FadeIn(dflt))
        self.play(*[s.animate.set_stroke(opacity=0.15) for s in segs], FadeIn(sdots, lag_ratio=0.02), run_time=2)
        got = T(f"{len(sdots)} points reach TensorBoard", 22, LOG).next_to(dflt, DOWN, buff=0.35, aligned_edge=LEFT)
        self.play(FadeIn(got))
        self.wait(2)
        code2 = CodeBlock('self.log_dict({"train_loss": loss},\n    on_step=False, on_epoch=True)', 16)
        code2.move_to(code1, aligned_edge=UL)
        code2.color_regex(r"self\.log_dict", LOG).color_regex(r"on_epoch=True", FIRE)
        # VO: "on_epoch=True: average every step of the epoch into one honest number."
        self.play(FadeTransform(code1, code2), FadeOut(VGroup(note, knob, dflt, got, sdots)),
                  *[s.animate.set_stroke(opacity=1) for s in segs])
        means = DATA["epoch_train"][:3]
        dots = VGroup(*[Dot(ax.c2p(n / 6 + e * n / 3, float(means[e])), radius=0.09, color=LOG) for e in range(3)])
        self.play(*[sg.animate.stretch_to_fit_height(0.02).set_y(ax.c2p(0, float(means[e]))[1])
                    for e, sg in enumerate(segs)], run_time=1.5)
        self.play(*[ReplacementTransform(sg, d) for sg, d in zip(segs, dots)], run_time=1)
        line = VMobject(stroke_color=LOG, stroke_width=3).set_points_as_corners([d.get_center() for d in dots])
        vals = DATA["epoch_val"][:3]
        vdots = VGroup(*[Dot(ax.c2p(n / 6 + e * n / 3, float(vals[e])), radius=0.07, color=VAL) for e in range(3)])
        vline = DashedLine(*[d.get_center() for d in vdots[:2]], color=VAL)
        vline2 = DashedLine(*[d.get_center() for d in vdots[1:]], color=VAL)
        leg = VGroup(M("train_loss (epoch mean)", 13, LOG), M("val_loss", 13, VAL)).arrange(DOWN, aligned_edge=LEFT)
        leg.next_to(ax.c2p(n * 0.55, 2.2), RIGHT, buff=0)
        self.play(Create(line), FadeIn(vdots), Create(vline), Create(vline2), FadeIn(leg))
        self.wait(1.5)

        # ---- where it goes: the folder tree
        lg = CodeBlock('logger = TensorBoardLogger("tb_logs",\n    name="mnist_model_v0")', 15)
        lg.color_regex("TensorBoardLogger", LOG).next_to(code2, DOWN, buff=0.6, aligned_edge=LEFT)
        tree = CodeBlock('tb_logs/\n  mnist_model_v0/\n    version_0/\n      events.out.tfevents...\n'
                         '      hparams.yaml\n    version_1/', 15, GREY_B)
        tree.next_to(lg, DOWN, buff=0.45, aligned_edge=LEFT)
        tree[5].set_color(FIRE)
        # VO: "The logger writes those numbers to disk. Every run gets its own version folder."
        self.play(FadeIn(lg))
        self.play(LaggedStart(*[FadeIn(l, shift=0.1 * RIGHT) for l in tree[:5]], lag_ratio=0.2))
        self.play(FadeIn(tree[5], shift=0.1 * RIGHT), Indicate(tree[5], color=FIRE))
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)))

        # ---- images: what the model actually sees (folder 8 adds random flips)
        img_code = CodeBlock('if batch_idx % 100 == 0:\n    grid = torchvision.utils.make_grid(x[:8].view(-1, 1, 28, 28))\n'
                             '    self.logger.experiment.add_image("mnist_images", grid, self.global_step)', 15)
        img_code.color_regex("add_image", LOG).to_edge(UP, buff=0.5)
        aug = M("transforms: RandomVerticalFlip(), RandomHorizontalFlip()", 14, DAT)
        aug.next_to(img_code, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(img_code), FadeIn(aug))
        tl = Line([-6.2, -1.6, 0], [6.2, -1.6, 0], color=GREY_C)
        tx = [-5.4 + 2.16 * i for i in range(6)]
        ticks = VGroup(*[VGroup(Line([x, -1.7, 0], [x, -1.5, 0], color=GREY_C),
                                M(str(100 * i), 13, GREY_B).move_to([x, -1.95, 0])) for i, x in enumerate(tx)])
        gs = M("global_step", 13, GREY_B).next_to(tl, DOWN, buff=0.6).to_edge(RIGHT, buff=0.6)
        self.play(Create(tl), FadeIn(ticks), FadeIn(gs))
        rng = np.random.default_rng(3)
        strips = []
        for i, x in enumerate(tx):
            ims = []
            for j in range(8):
                a = DATA["digits"][(i * 8 + j) % 64]
                if i == 3 and j == 5:
                    a = DATA["six"][::-1, ::-1]          # both flips = 180 degrees
                else:
                    if rng.random() < 0.5:
                        a = a[::-1, :]
                    if rng.random() < 0.5:
                        a = a[:, ::-1]
                ims.append(digit_img(a, 0.24))
            s = Group(*ims).arrange(RIGHT, buff=0.02).move_to([x, -1.05, 0])
            strips.append(s)
        # VO: "add_image pins a snapshot of the batch to the timeline: what the model actually sees."
        self.play(LaggedStart(*[FadeIn(s, shift=0.2 * UP) for s in strips], lag_ratio=0.25, run_time=2.5))
        self.wait()
        focus = strips[3]
        others = Group(*[s for k, s in enumerate(strips) if k != 3], tl, ticks, gs, img_code, aug)
        self.play(FadeOut(others), focus.animate.scale(5.5).move_to(0.6 * UP), run_time=1.5)
        six = focus[5]
        b = SurroundingRectangle(six, color=FIRE, buff=0.05, stroke_width=3)
        l1 = M("label: 6", 20, FIRE).next_to(focus, DOWN, buff=0.4)
        l2 = T("both flips at once rotate it 180 degrees: is that a 9?", 26, GREY_A).next_to(l1, DOWN, buff=0.3)
        # VO: "One flip in four does both, and a six turns into a nine. TensorBoard is how you'd catch that."
        self.play(Create(b), FadeIn(l1))
        self.play(FadeIn(l2))
        self.wait(2.5)
        cmd = M("$ tensorboard --logdir tb_logs", 24, LOG).to_edge(DOWN, buff=0.4)
        self.play(FadeOut(l2), FadeIn(cmd))
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)))


# =============================================================== 5. callbacks
class S5_Callbacks(Scene):
    def construct(self):
        q = T("How do you add behaviour without opening the machine?", 32).move_to(UP * 0.3)
        # VO: "Last question: you want to print, log, stop early. Do you have to open the Trainer?"
        self.play(FadeIn(q))
        self.wait(2)
        self.play(FadeOut(q))

        C, R = np.array([-3.7, 0.55, 0]), 2.1
        dial = make_dial(C, R)
        machine = RoundedRectangle(corner_radius=0.3, width=5.4, height=5.4, stroke_color=ENG, stroke_width=2)
        machine.move_to(C)
        mlabel = M("pl.Trainer", 16, ENG).next_to(machine.get_corner(UL), DR, buff=0.18)
        theta = ValueTracker(0)
        dot = Dot(clock_point(C, R, 0), radius=0.1, color=FIRE)
        dot.add_updater(lambda m: m.move_to(clock_point(C, R, theta.get_value())))
        HV, HT = 326, 344
        sockets = VGroup(*[Circle(0.09, stroke_color=WHITE, stroke_width=2).move_to(clock_point(C, R, d))
                           for d in (HV, HT)])
        ep = T("epoch 0", 24).move_to(C + 0.3 * UP)
        hook = M("", 14).move_to(C + 0.25 * DOWN)
        self.play(Create(machine), FadeIn(mlabel), Create(dial), FadeIn(dot), FadeIn(ep))
        # VO: "The machine already has sockets: named moments, called hooks."
        self.play(LaggedStart(*[GrowFromCenter(s) for s in sockets], lag_ratio=0.3))
        self.add(hook)

        chips = VGroup(chip("MyPrintingCallback"), chip("EpochMetricsCallback"),
                       chip("NN.on_train_epoch_end", SCI), chip("EarlyStopping"))
        chips.arrange(RIGHT, buff=0.3).to_edge(DOWN, buff=0.35)
        console = M("", 13, GREY_B).move_to([C[0], -2.65, 0])
        self.play(FadeIn(chips, lag_ratio=0.2))
        self.add(console)

        vals = DATA["epoch_val"]
        ax = Axes(x_range=[0, 20, 5], y_range=[0.09, 0.27, 0.06], x_length=5.4, y_length=3.0, tips=False,
                  axis_config={"color": GREY_C, "stroke_width": 1.5})
        ax.move_to([3.6, 1.6, 0])
        axl = M("val_loss", 14, VAL).next_to(ax.y_axis, UP, buff=0.12)
        axx = M("epoch", 13, GREY_B).next_to(ax.x_axis, DOWN, buff=0.15).align_to(ax.x_axis, RIGHT)
        pat_l = M("patience", 15, GREY_B).move_to([1.6, -0.7, 0])
        pdots = VGroup(*[Circle(0.12, stroke_color=FIRE, stroke_width=2, fill_color=FIRE, fill_opacity=1)
                         for _ in range(3)]).arrange(RIGHT, buff=0.18).next_to(pat_l, RIGHT, buff=0.3)
        status = M("", 14, GREY_B).move_to([3.6, -1.35, 0])
        self.play(Create(ax), FadeIn(axl), FadeIn(axx), FadeIn(pat_l), FadeIn(pdots))
        self.add(status)
        best_line = DashedLine(ax.c2p(0, 0.25), ax.c2p(20, 0.25), color=DAT, stroke_width=2).set_opacity(0)
        self.add(best_line)

        def say(mob, s, color=GREY_B, size=14, rt=0.2, pos=None):
            return swap_text(self, mob, M(s, size, color), rt, pos)

        # on_train_start
        hook = say(hook, "on_train_start", FIRE, pos=C + 0.25 * DOWN)
        self.play(Indicate(chips[0], color=FIRE), run_time=0.8)
        console = say(console, "Starting to train!", GREY_B, 13, pos=[C[0], -2.65, 0])

        best, wait, prev_pt, curve = np.inf, 0, None, VGroup()
        self.add(curve)
        for e in range(len(vals)):
            slow = e == 0
            theta.set_value(0)
            ep.become(T(f"epoch {e}", 24).move_to(C + 0.3 * UP))
            self.play(theta.animate.set_value(HV), run_time=2.4 if slow else 0.45, rate_func=linear)
            # on_validation_epoch_end
            hook.become(M("on_validation_epoch_end", 14, FIRE).move_to(C + 0.25 * DOWN))
            v = float(vals[e])
            p = Dot(ax.c2p(e, v), radius=0.05, color=VAL)
            seg = [Line(prev_pt.get_center(), p.get_center(), color=VAL, stroke_width=2)] if prev_pt else []
            self.play(Flash(sockets[0], color=FIRE, line_length=0.15, run_time=0.4),
                      Indicate(chips[1], color=FIRE), FadeIn(p), *[Create(s) for s in seg],
                      run_time=0.9 if slow else 0.35)
            curve.add(p, *seg)
            prev_pt = p
            self.play(theta.animate.set_value(HT), run_time=0.5 if slow else 0.12, rate_func=linear)
            # on_train_epoch_end: callbacks, then your module, then monitoring callbacks
            hook.become(M("on_train_epoch_end", 14, FIRE).move_to(C + 0.25 * DOWN))
            if slow:
                nums = VGroup(*[T(str(i + 1), 22, FIRE).next_to(chips[j], UP, buff=0.12)
                                for i, j in enumerate((1, 2, 3))])
                # VO: "Who goes first? Callbacks, then your module, then callbacks that watch a
                #      metric, so EarlyStopping sees what your module just logged."
                self.play(Flash(sockets[1], color=FIRE, line_length=0.15), run_time=0.4)
                for i, j in enumerate((1, 2, 3)):
                    self.play(Indicate(chips[j], color=FIRE), FadeIn(nums[i]), run_time=0.7)
                    if j == 2:
                        console = say(console, "Epoch 0 finished. Current learning rate: 0.001", GREY_B, 13)
                self.wait(1.5)
                self.play(FadeOut(nums))
            else:
                self.play(Flash(sockets[1], color=FIRE, line_length=0.15), Indicate(chips[1], color=FIRE),
                          run_time=0.3)
                self.play(Indicate(chips[2], color=FIRE), run_time=0.25)
                self.play(Indicate(chips[3], color=FIRE), run_time=0.25)
                console.become(M(f"Epoch {e} finished. Current learning rate: 0.001", 13, GREY_B)
                               .move_to([C[0], -2.65, 0]))
            # EarlyStopping(monitor="val_loss"), default patience=3, mode="min"
            if v < best:
                best, wait = v, 0
                msg, col = f"new best {v:.4f}, patience refills", DAT
                y = ax.c2p(0, v)[1]
                self.play(best_line.animate.set_opacity(1).set_y(y), run_time=0.3)
            else:
                wait += 1
                msg, col = f"no improvement ({wait}/3)", GREY_B
            self.play(*[d.animate.set_fill(opacity=1 if i < 3 - wait else 0) for i, d in enumerate(pdots)],
                      run_time=0.2)
            status.become(M(msg, 14, col).move_to([3.6, -1.35, 0]))
            if wait >= 3:
                status.become(M("trainer.should_stop = True", 15, FIRE).move_to([3.6, -1.35, 0]))
                self.play(Indicate(status, color=FIRE), run_time=0.8)
                break
            self.play(theta.animate.set_value(360), run_time=0.4 if slow else 0.1, rate_func=linear)

        # on_train_end
        self.wait(0.5)
        hook = say(hook, "on_train_end", FIRE)
        self.play(Indicate(chips[0], color=FIRE), run_time=0.8)
        console = say(console, "Training is done.", GREY_B, 13)
        tag = T(f"stopped after epoch {e}: the best val_loss was epoch {int(np.argmin(vals[:e + 1]))}",
                22, GREY_A).move_to([3.6, -2.0, 0])
        tag.scale_to_fit_width(min(tag.width, 6.2))
        self.play(FadeIn(tag))
        self.wait(3)
        dot.clear_updaters()
        self.play(FadeOut(Group(*self.mobjects)))


# ================================================================== 6. recap
TRAIN_PY = '''from model import NN
from dataset import MnistDataModule
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import config
logger = TensorBoardLogger("tb_logs", name="mnist_model_v0")
model = NN(
    input_size=config.INPUT_SIZE,
    learning_rate=config.LEARNING_RATE,
    num_classes=config.NUM_CLASSES,
)
dm = MnistDataModule(
    data_dir=config.DATA_DIR,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
)
trainer = pl.Trainer(
    logger=logger,
    accelerator=config.ACCELERATOR,
    max_epochs=config.NUM_EPOCHS,
    precision=config.PRECISION,
    callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")],
)
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)'''


class S6_Recap(Scene):
    def construct(self):
        code = CodeBlock(TRAIN_PY, 13).to_edge(LEFT, buff=0.5)
        cats = "SDFLW" + "L" + "SSSSS" + "DDDDD" + "EEEEEFE" + "EEE"
        colmap = {"S": SCI, "D": DAT, "E": ENG, "F": FIRE, "L": LOG, "W": WHITE}
        for l, c in zip(code, cats):
            l.set_color(colmap[c])
        code.part(17, "logger=logger").set_color(LOG)
        code.color_regex(r"config\.\w+", WHITE)
        for i in (23, 24, 25):
            code.part(i, "model").set_color(SCI)
            code.part(i, "dm").set_color(DAT)
        cards = VGroup(box("model.py", SCI, 2.8, 0.6, 16), box("dataset.py", DAT, 2.8, 0.6, 16),
                       box("callbacks.py", FIRE, 2.8, 0.6, 16), box("config.py", WHITE, 2.8, 0.6, 16),
                       box("tb_logs/", LOG, 2.8, 0.6, 16)).arrange(DOWN, buff=0.3).to_edge(RIGHT, buff=0.8)
        head = M("train.py", 16, GREY_B).next_to(code, UP, buff=0.2, aligned_edge=LEFT)
        # VO: "Put it back together. train.py is now a table of contents."
        self.play(FadeIn(head), FadeIn(code, lag_ratio=0.03), run_time=2.5)
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * LEFT) for c in cards], lag_ratio=0.2), run_time=1.5)
        self.wait(2)
        tag = VGroup(T("science in the module,", 26, SCI), T("data in the data module,", 26, DAT),
                     T("engineering in the Trainer", 26, ENG)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        tag.next_to(cards, DOWN, buff=0.4).to_edge(RIGHT, buff=0.5)
        if tag.get_bottom()[1] < -3.8:
            tag.to_edge(DOWN, buff=0.25)
        # VO: "Your science in the module, your data in the data module, and the engineering in the Trainer."
        self.play(LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in tag], lag_ratio=0.5), run_time=2.5)
        self.wait(3)
