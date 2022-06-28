# %%
from functools import partial
import pickle
import random
from tqdm import tqdm
import itertools
import numpy as np
import copy
import collections
import heapq
import math
from typing import Dict, Optional
from PIL import Image, ImageDraw, ImageFont
from ast import operator
from dataclasses import dataclass, field
from string import ascii_letters, ascii_lowercase


# %%
@dataclass
class Layout:
    grid: dict = field(default_factory=dict)
    machines: Dict[str, "Machine"] = field(default_factory=dict)

    def get(self, x, y):
        return self.grid[(x, y)]

    def set(self, x, y, val):
        self.grid[(x, y)] = val

    def get_neighbors(self, x, y):
        for i, j in (1, 0), (-1, 0), (0, 1), (0, -1):
            if (x + i, y + j) in self.grid:
                yield (x + i, y + j)

    def find(self, val):
        for x, y in self.grid:
            if self.grid[(x, y)] == val:
                return x, y
        raise ValueError(f"{val} not found")

    @property
    def width(self):
        return max(x for x, y in self.grid) + 1

    @property
    def height(self):
        return max(y for x, y in self.grid) + 1

    def crossable(self, x, y):
        if (x, y) not in self.grid:
            return False
        machine_outputs = [m.output_pos for m in self.machines.values()]
        tile = self.get(x, y)
        return tile == FLOOR or tile == ENTRANCE or (x, y) in machine_outputs

    def could_fit_machine(self, machine: "Machine"):
        assert machine.output_pos in machine.domain
        assert len(set(machine.domain)) == len(machine.domain)
        for x, y in machine.domain:
            if not self.crossable(x, y):
                return False
            if (x, y) == self.find(ENTRANCE):
                return False
        return True

    def fit_machine(self, machine: "Machine"):
        assert self.could_fit_machine(machine)
        for x, y in machine.domain:
            self.set(x, y, machine)
        self.machines[machine.name] = machine

    def remove_machine(self, machine: "Machine"):
        for x, y in machine.domain:
            self.set(x, y, FLOOR)
        del self.machines[machine.name]

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class Machine:
    name: str
    domain: list
    orientaion: int
    output_pos: tuple


FLOOR = "floor"
ITEM_COUNT = 3
ENTRANCE = "entrance"


def parse(token):
    if token == ".":
        return FLOOR
    elif token == "@":
        return ENTRANCE
    elif token in ascii_letters:
        return token
    else:
        raise ValueError(token)


# %%
def hamming_dist(x_0, y_0, x_1, y_1):
    return abs(x_0 - x_1) + abs(y_0 - y_1)


class PathNotFound(Exception):
    pass


@dataclass
class Canvas:
    width: int
    height: int
    data: dict = field(default_factory=dict)
    worker_loc: Optional[tuple] = None

    def add_floor(self, x, y):
        self.data[(x, y)] = FLOOR

    def add_entrance(self, x, y):
        self.data[(x, y)] = ENTRANCE

    def add_machine(self, machine: Machine):
        for x, y in machine.domain:
            self.data[(x, y)] = machine

    def fill_floor(self):
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.data:
                    self.add_floor(x, y)

    def add_worker(self, x, y):
        self.worker_loc = (x, y)

    @staticmethod
    def from_layout(layout: Layout):
        canvas = Canvas(layout.width, layout.height)
        for x, y in layout.grid:
            if layout.get(x, y) == FLOOR:
                canvas.add_floor(x, y)
            elif layout.get(x, y) == ENTRANCE:
                canvas.add_entrance(x, y)
            elif isinstance(layout.get(x, y), Machine):
                canvas.add_machine(layout.get(x, y))
        return canvas

    def __str__(self):
        s = ""
        for y in range(self.height):
            for x in range(self.width):
                item = self.data.get((x, y), " ")
                if (x, y) == self.worker_loc:
                    item = "&"
                elif isinstance(item, Machine):
                    if (x, y) == item.output_pos:
                        item = item.name
                    else:
                        item = item.name.lower()
                elif item == FLOOR:
                    item = "."
                elif item == ENTRANCE:
                    item = "@"
                s += item
            s += "\n"
        return s


@dataclass
class Node:
    x: int
    y: int
    cost: int = 0
    heuristic: int = 0

    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


def find_path(layout: Layout, start, goal):
    parents = {}
    frontier = [Node(start[0], start[1], 0, hamming_dist(*start, *goal))]
    visited = set()
    visited.add(start)
    goal_found = False
    while not goal_found:
        if not frontier:
            raise PathNotFound
        node = heapq.heappop(frontier)
        for adj_pos in layout.get_neighbors(node.x, node.y):
            if goal in visited:
                goal_found = True
                break
            elif layout.crossable(*adj_pos) and (adj_pos not in visited):
                new_node = Node(
                    adj_pos[0],
                    adj_pos[1],
                    node.cost + 1,
                    hamming_dist(*adj_pos, *goal),
                )
                heapq.heappush(frontier, new_node)
                parents[adj_pos] = (node.x, node.y)
                visited.add(adj_pos)
    path = []
    track = goal
    while track != start:
        path.append(track)
        track = parents[track]
    return path[::-1]


@dataclass
class Factory:
    layout: Layout
    production: dict
    worker_loc: tuple
    assets: list = field(default_factory=list)
    worker_holding_item: Optional[str] = None
    worker_path: list = field(default_factory=list)
    clock: int = 0
    images: list = field(default_factory=list)
    deliveries: collections.Counter = field(default_factory=collections.Counter)
    idle_time: int = 0

    @property
    def saturation(self):
        return 1 - (self.idle_time / self.clock)

    def tick(self):
        for machine_name, freq in self.production.items():
            rate = 1 / freq
            if random.random() < rate:
                machine = self.layout.machines[machine_name]
                self.assets.append((machine.name, machine.output_pos))

        # dump item
        if self.worker_loc == self.layout.find(ENTRANCE) and self.worker_holding_item:
            self.deliveries[self.worker_holding_item] += 1
            self.worker_holding_item = None

        # travel
        if self.worker_path:
            self.worker_loc = self.worker_path.pop(0)
            if self.assets and self.worker_loc == self.assets[0][1]:
                self.worker_holding_item = self.assets.pop(0)[0]

        # plan to exit
        elif self.worker_holding_item:
            self.worker_path = find_path(
                self.layout, self.worker_loc, self.layout.find(ENTRANCE)
            )

        # plan to item
        elif self.assets:
            goal = self.assets[0][1]
            self.worker_path = find_path(self.layout, self.worker_loc, goal)

        # idle
        else:
            self.idle_time += 1

        self.clock += 1

    def reset(self):
        self.clock = 0
        self.worker_loc = self.layout.find(ENTRANCE)
        self.worker_holding_item = False
        self.worker_path = []
        self.assets = []
        self.images = []

    def __str__(self):
        canvas = Canvas(width=self.layout.width, height=self.layout.height)
        canvas.add_worker(*self.worker_loc)
        for m in self.layout.machines.values():
            canvas.add_machine(m)
        for x, y in self.layout.grid:
            if self.layout.get(x, y) == FLOOR:
                canvas.add_floor(x, y)
        return str(canvas)

    def render(self):
        font = ImageFont.truetype("courier.ttf", 20)
        width, height = font.getsize_multiline(str(self))
        im = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(im)
        draw.multiline_text((0, 0), str(self), font=font, fill="white")
        self.images.append(im)

    def save(self, path):
        self.images[0].save(
            path,
            save_all=True,
            append_images=self.images[1:],
            optimize=False,
            duration=60,
            loop=1,
        )
        self.images.clear()

    def copy(self):
        return copy.deepcopy(self)


def parse_factory(path):
    with open(path) as f:
        lines = f.read().splitlines()
        production = {}
        for i in range(ITEM_COUNT):
            item, prod_str = lines[i].split(":")
            production[item] = float(prod_str)

        lo = Layout()
        for y, line in enumerate(lines[ITEM_COUNT:]):
            for x, token in enumerate(line):
                token = parse(token)
                lo.set(x, y, token)

        for x, y in lo.grid:
            curr_token = lo.get(x, y)
            if isinstance(curr_token, str) and curr_token in ascii_letters:
                machine_name: str = lo.get(x, y).upper()
                stack = [(x, y)]
                visited = {(x, y)}
                domain = [(x, y)]
                while stack:
                    top = stack.pop()
                    for i, j in lo.get_neighbors(*top):
                        if isinstance(lo.get(i, j), Machine):
                            continue
                        if (
                            lo.get(i, j).upper() == machine_name
                            and (i, j) not in visited
                        ):
                            domain.append((i, j))
                            stack.append((i, j))
                            visited.add((i, j))
                output = next(
                    ((x, y) for x, y in domain if lo.get(x, y) == machine_name)
                )
                machine = Machine(machine_name, domain, 0, output)
                for (x, y) in domain:
                    lo.set(x, y, machine)
                lo.machines[machine_name] = machine

        worker_loc = lo.find(ENTRANCE)

    return Factory(layout=lo, production=production, worker_loc=worker_loc)


# %%
def rotate_machine(machine: Machine, quarter):
    angle = quarter * 90
    new_domain = []
    center = machine.output_pos
    for x, y in machine.domain:
        new_x, new_y = rotate(x, y, center, angle)
        new_x = np.rint(new_x).astype(int)
        new_y = np.rint(new_y).astype(int)
        new_domain.append((new_x, new_y))
    return Machine(
        machine.name, new_domain, machine.orientaion + quarter, machine.output_pos,
    )


def rotate(x, y, center, angle):
    x, y = x - center[0], y - center[1]
    x, y = rotate_point(x, y, angle)
    return x + center[0], y + center[1]


def rotate_point(x, y, angle):
    angle = math.radians(angle)
    x, y = (
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle),
    )
    return x, y


def get_rotated_machines(machine):
    for i in range(4):
        yield rotate_machine(machine, i)


def shift_machine(machine: Machine, x_shift, y_shift):
    new_domain = []
    for x, y in machine.domain:
        new_x, new_y = x + x_shift, y + y_shift
        new_domain.append((new_x, new_y))
    new_output_pos = machine.output_pos[0] + x_shift, machine.output_pos[1] + y_shift
    return Machine(machine.name, new_domain, machine.orientaion, new_output_pos)


def get_shift_ranges(machine, layout):
    min_x = min(machine.domain, key=lambda x: x[0])[0]
    max_x = max(machine.domain, key=lambda x: x[0])[0]
    min_y = min(machine.domain, key=lambda x: x[1])[1]
    max_y = max(machine.domain, key=lambda x: x[1])[1]
    x_shift_range = range(-min_x, layout.width - max_x)
    y_shift_range = range(-min_y, layout.height - max_y)
    for x_shift in x_shift_range:
        for y_shift in y_shift_range:
            if (x_shift, y_shift) == (0, 0):
                continue
            yield (x_shift, y_shift)


def get_shifted_machines(machine, layout):
    for x_shift, y_shift in get_shift_ranges(machine, layout):
        yield shift_machine(machine, x_shift, y_shift)


def get_machine_rotated_and_shifted(machine, layout):
    for machine_rotated in get_rotated_machines(machine):
        for machine_shifted in get_shifted_machines(machine_rotated, layout):
            yield machine_shifted


def get_machines_rotated_and_shifted(machines, layout):
    yield from itertools.product(
        *[get_machine_rotated_and_shifted(m, layout) for m in machines]
    )


# %%
def benchmark_factory(factory: Factory, ticks=1000):
    for _ in range(ticks):
        factory.tick()
    return sum(factory.deliveries.values()) / ticks


# %%
orig_factory = parse_factory("layouts/1.txt")
machines_to_add = orig_factory.layout.machines.values()
base_layout = orig_factory.layout.copy()
for m in machines_to_add:
    base_layout.remove_machine(m)

orig_factory.reset()
for _ in range(1000):
    orig_factory.tick()
    orig_factory.render()
orig_factory.save("gifs/orig.gif")
print("saturation:", orig_factory.saturation)

# %%
combos = get_machines_rotated_and_shifted(machines_to_add, base_layout)
combos = list(combos)
random.shuffle(combos)

# %%
factories = []
results = []
good_layouts = []
for i, combo in tqdm(list(enumerate(combos))):
    lo = base_layout.copy()
    valid = True
    for m in combo:
        try:
            lo.fit_machine(m)
        except:
            valid = False
            break
    if not valid:
        continue
    orig_factory = orig_factory.copy()
    orig_factory.layout = lo
    orig_factory.reset()
    try:
        for m in combo:
            find_path(lo, orig_factory.worker_loc, m.output_pos)
    except PathNotFound:
        valid = False

    if not valid:
        continue
    else:
        good_layouts.append(lo)

    factory = Factory(
        layout=lo,
        production=orig_factory.production,
        worker_loc=orig_factory.worker_loc,
    )
    benchmark_result = benchmark_factory(factory)
    factories.append(factory)
    results.append(benchmark_result)

    if len(results) % 100 == 0:
        best_factory = factories[np.argmax(results)]
        best_factory.reset()
        for _ in range(300):
            best_factory.tick()
            best_factory.render()
        best_factory.save("gifs/" + str(len(results)) + ".gif")
        # save result to pickle
        with open(f"result.pickle", "wb") as f:
            pickle.dump(results, f)

# %%
best_factory = factories[np.argmax(results)]
best_factory.reset()
for _ in range(300):
    best_factory.tick()
    best_factory.render()
best_factory.save("gifs/" + str(len(results)) + ".gif")
# save result to pickle
with open(f"result.pickle", "wb") as f:
    pickle.dump(results, f)


# %%
results_maxes = [max(results[: i + 1]) for i in range(len(results))]

# %%
import seaborn as sns
from scipy import interpolate

x = np.arange(len(results))
y = np.array(results_maxes) * 60

plt = sns.lineplot(x, y)
plt.set_xlim(0, len(results))
plt.set_ylim(0, 5)
plt.set_xlabel("AI Optimization Iteration")
plt.set_ylabel("Item Throughput (per minute)")

# %%
# orig_factory.reset()
# benchmark_factory(orig_factory)
# %%

# %%
orig_factory = parse_factory("layouts/1.txt")
orig_factory.reset()
for _ in range(1000):
    orig_factory.tick()
orig_deliveries = orig_factory.deliveries
best_deliveries = best_factory.deliveries

# %%
import pandas as pd

# %%
data = []
for k, v in orig_deliveries.items():
    data.append({"SKU": k, "item count": v, "name": "before"})
for k, v in best_deliveries.items():
    data.append({"SKU": k, "item count": v, "name": "after"})
df = pd.DataFrame(data=data)
print(df)
# %%
sns.barplot(x="SKU", y="item count", hue="name", data=df)

# %%
tips = sns.load_dataset("tips")
# %%
facs = random.choices(factories, k=5)
for i, f in enumerate(facs):
    f.reset()
    for _ in range(300):
        f.tick()
        f.render()
    f.save(f"gifs/more_{i}.gif")
# %%
results_maxes[-1] * 60