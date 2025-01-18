from abc import ABC
from enum import Enum
from typing import Iterator, Optional, Any, overload
import pygame as pg
import pygame_gui as gui
from numpy.random import choice
import json
import networkx as nx
import numpy as np
from numpy.linalg import norm
import os



def get_passability_graph(bitmap: list[list[Any]]) -> nx.Graph:
    graph = nx.Graph()
    for y, row in enumerate(bitmap):
        for x, cell in enumerate(row):
            if cell:
                continue
            graph.add_node((x, y))
            if (x-1, y) in graph:
                graph.add_edge((x-1, y), (x, y))
            if (x, y-1) in graph:
                graph.add_edge((x, y-1), (x, y))
    nx.set_edge_attributes(graph, 1, "weight")
    return graph


@overload
def find_path(source: list[list[Any]], start: tuple[int, int], finish: tuple[int, int]) -> list[tuple[int, int]] | None:
    ...

@overload
def find_path(source: nx.Graph, start: tuple[int, int], finish: tuple[int, int]) -> list[tuple[int, int]] | None:
    ...


def find_path(source: nx.Graph | list[list[Any]], start: tuple[int, int], finish: tuple[int, int]) -> list[tuple[int, int]] | None:
    if isinstance(source, nx.Graph):
        graph = source
    else:
        graph = get_passability_graph(source)
    
    nx.set_node_attributes(graph, {node : norm(np.array(finish)-np.array(node)) for node in graph.nodes}, 'weight')
    nx.set_node_attributes(graph, np.inf, 'path')
    nx.set_node_attributes(graph, {start : 0.}, 'path')
    nx.set_node_attributes(graph, {start : graph.nodes[start]['weight']}, 'cost')

    front: dict[tuple, tuple | None] = {start : None}
    visited = {}

    while front:
        new_node = min(front, key = lambda node: graph.nodes[node]['cost'])
        visited[new_node] = front[new_node]
        front.pop(new_node)
        
        if new_node == finish:
            break
        
        for n in graph.neighbors(new_node):
            if n in visited:
                continue
            if graph.nodes[new_node]['path'] + graph.edges[n, new_node]['weight'] < graph.nodes[n]['path'] or n not in front:
                nx.set_node_attributes(graph, {n : graph.nodes[new_node]['path'] + graph.edges[n, new_node]['weight']}, 'path')
                nx.set_node_attributes(graph, {n : graph.nodes[n]['path'] + graph.nodes[n]['weight']}, 'cost')
                front[n] = new_node
    else:
        return None
    
    path = [finish]
    while visited[path[-1]]:
        path.append(visited[path[-1]])
    return path[::-1]


class Textures:
    
    balls = {}
    
    @staticmethod
    def load():
        if os.path.exists('data/textures'):
            images = os.listdir('data/textures')
            Textures.balls = {image.split('.')[0] : pg.image.load(f'data/textures/{image}') for image in images}


class Colors(Enum):
    white = (255, 255, 255)
    black = (0, 0, 0)
    gray = (23, 23, 23)
    lightgray = (46, 46, 46)
    red = (169, 30, 13)
    green = (93, 167, 33)
    blue = (46, 63, 186)    
    orange = (238, 96, 0)
    cyan = (9, 213, 255)
    yellow = (206, 210, 0)
    purple = (95, 12, 127)


class CellStages(Enum):
    empty = 0
    seed = 1
    occupied = 2
    

class CellColors(Enum):
    red = Colors.red.value
    green = Colors.green.value
    blue = Colors.blue.value
    orange = Colors.orange.value
    cyan = Colors.cyan.value
    yellow = Colors.yellow.value
    purple = Colors.purple.value


class Effect(ABC):
    
    def __init__(self, duration: int, loop: bool) -> None:
        super().__init__()
        self.tick = 0
        self.duration = duration
        self.ended = False
        self.loop = loop
    
    def apply(self, image: pg.Surface) -> pg.Surface:
        ...
    
    def update(self) -> bool:
        ...


class ScalingEffect(Effect):
    
    def __init__(self, start_scale: float = 1, finish_scale: float = 1, duration: int = 60, loop: bool = True) -> None:
        super().__init__(duration, loop)
        self.start_scale = start_scale
        self.finish_scale = finish_scale
    
    def apply(self, image: pg.Surface) -> pg.Surface:
        canvas = pg.Surface(image.get_size(), pg.SRCALPHA)
        scale = (self.finish_scale - self.start_scale) / self.duration * self.tick + self.start_scale
        transformed = pg.transform.scale_by(image, scale)
        rect = transformed.get_rect()
        rect.center = canvas.get_rect().center
        canvas.blit(transformed, rect)
        
        self.tick += 1
        if self.tick > self.duration:
            self.ended = True
        return canvas
    
    def update(self) -> bool | Effect:
        if not self.loop:
            return not self.ended
        else:
            return ScalingEffect(self.finish_scale, self.start_scale, self.duration) if self.ended else True


class ColorEffect(Effect):
    
    def __init__(self, colorkey: tuple[int, int, int], start_color: tuple[int, int, int], finish_color: tuple[int, int, int], duration: int = 60, loop: bool = True) -> None:
        super().__init__(duration, loop)
        self.start_color = start_color
        self.finish_color = finish_color
        self.colorkey = colorkey
    
    def apply(self, image: pg.Surface) -> pg.Surface:
        canvas = pg.Surface(image.get_size(), pg.SRCALPHA)
        color = tuple(int((self.finish_color[i] - self.start_color[i]) / self.duration * self.tick + self.start_color[i]) for i in range(3))
        canvas.fill(color)
        image.set_colorkey(self.colorkey)
        canvas.blit(image, (0, 0))
        
        self.tick += 1
        if self.tick > self.duration:
            self.ended = True
        return canvas
    
    def update(self) -> bool | Effect:
        if not self.loop:
            return not self.ended
        else:
            return ColorEffect(self.colorkey, self.finish_color, self.start_color, self.duration) if self.ended else True


class Cell(pg.sprite.Sprite):
    
    def __init__(self, coords: tuple[int, int], pos: tuple[int, int] = (0, 0), size: int = 72) -> None:
        super().__init__()
        self.stage: CellStages = CellStages.empty
        self.color: Optional[CellColors] = None
        self.bg_color = Colors.gray.value
        self.bg_hovered_color = Colors.lightgray.value
        self.coords = coords
        self.size = size
        self.hovered = False
        self.fg_effects: list[Effect] = []
        self.bg_effects: list[Effect] = []
        self.image = self.update()
        self.rect = self.image.get_rect()
        self.rect.topleft = pos

    
    def grow(self):
        if self.stage is CellStages.seed:
            self.stage = CellStages.occupied
            self.fg_effects.append(ScalingEffect(9/25, 1, 20, loop=False))
    
    
    def select(self):
        self.fg_effects.append(ScalingEffect(finish_scale=0.7, duration=20))


    def deselect(self):
        for e in self.fg_effects:
            if isinstance(e, ScalingEffect):
                self.fg_effects.remove(e)
                break
    
    
    def update(self) -> pg.Surface:
        width, height = self.size, self.size
        bg = pg.Surface((width, height))
        fg = pg.Surface((width, height), pg.SRCALPHA)
        pg.draw.rect(bg, self.bg_color if (not self.hovered) or any(isinstance(e, ColorEffect) for e in self.bg_effects) else self.bg_hovered_color, bg.get_rect(), border_radius=12)
        if self.stage is CellStages.seed:
            if self.color.name in Textures.balls: # type: ignore
                reduced = pg.transform.scale_by(Textures.balls[self.color.name], 9/25) # type: ignore
                rect = reduced.get_rect()
                rect.center = (width//2, height//2)
                fg.blit(reduced, rect)
            else:
                pg.draw.circle(fg, self.color.value, (width/2, height/2), 9) # type: ignore
        elif self.stage is CellStages.occupied:
            if self.color.name in Textures.balls: # type: ignore
                fg.blit(Textures.balls[self.color.name], (0, 0)) # type: ignore
            else:
                pg.draw.circle(fg, self.color.value, (width/2, height/2), 25) # type: ignore
            
        for effect in self.bg_effects:
            bg = effect.apply(bg)
        for effect in self.fg_effects:
            fg = effect.apply(fg)
        
        bg_effects = []
        for e in self.bg_effects:
            if state := e.update():
                bg_effects.append(e if state is True else state)
        self.bg_effects = bg_effects
        
        fg_effects = []
        for e in self.fg_effects:
            if state := e.update():
                fg_effects.append(e if state is True else state)
        self.fg_effects = fg_effects
            
        bg.blit(fg, (0, 0))
        self.image = bg
        return self.image
    
    
    def __bool__(self) -> bool:
        return self.stage is CellStages.occupied
    
    
    def __hash__(self) -> int:
        return hash(self.coords)
        

class Game:
    
    def __init__(self) -> None:
        pg.display.set_caption('Lines 98')
        self.score: int = 0
        self.turns: int = 0
        self.chosen: Cell | None = None
        self.cell_size: int = 72
        self.padding: int = 8
        self.load_stats()
        self.field: list[list[Cell]] = [[Cell(coords=(x, y), 
                                              pos=self.cells2pixels((x, y)), 
                                              size=self.cell_size) 
                                         for x in range(9)] for y in range(9)]
        self.sprite_group = pg.sprite.Group(self.all_cells()) # type: ignore
        for cell in self.spawn():
            cell.grow()
        self.spawn()
    
    
    def load_stats(self):
        with open('data/stats.json', 'r') as f:
            self.stats: dict = json.load(f)
    
    
    def save_stats(self):
        with open('data/stats.json', 'w') as f:
            json.dump(self.stats, f, indent=4)
    
    
    def select(self, cell: Cell | None):
        if self.chosen:
            self.chosen.deselect()
        self.chosen = cell
        if cell:
            cell.select()
        
        
    def get_cell(self, x: int, y: int) -> Cell:
        return self.field[y][x]
    
    
    def all_cells(self) -> Iterator[Cell]:
        for y in range(9):
            for cell in self.field[y]:
                yield cell
                
    
    def empty_cells(self) -> Iterator[Cell]:
        for y in range(9):
            for cell in self.field[y]:
                if cell.stage is CellStages.empty:
                    yield cell
                    
    
    def seed_cells(self) -> Iterator[Cell]:
        for y in range(9):
            for cell in self.field[y]:
                if cell.stage is CellStages.seed:
                    yield cell
    
    
    def spawn(self, n: int = 3) -> list[Cell]:
        n = min(n, 3 - len(list(self.seed_cells())))
        options = list(self.empty_cells())
        chosen: list[Cell] = list(choice(options, min(n, len(options)), replace=False)) # type: ignore
        for cell in chosen:
            cell.stage = CellStages.seed
            # cell.color = CellColors.green
            cell.color = choice(list(CellColors)) # type: ignore
            cell.fg_effects.append(ScalingEffect(0, 1, 20, loop=False))
        return chosen
    
    
    def get_path(self, start: Cell, finish: Cell) -> list[Cell] | None:
        bitmap = [[bool(self.field[y][x]) for x in range(9)] for y in range(9)]
        bitmap[start.coords[1]][start.coords[0]] = False
        path = find_path(bitmap, start.coords, finish.coords)
        return None if not path else [self.get_cell(x, y) for x, y in path]
    
    
    def swap(self, target: Cell, destination: Cell):
        destination.color, target.color = target.color, destination.color
        destination.stage, target.stage = target.stage, destination.stage
        
        
    def player_action(self, target: Cell, destination: Cell) -> bool:
        path = self.get_path(target, destination)
        if path:
            self.swap(target, destination)
            for cell in path:
                cell.bg_effects.append(ColorEffect(cell.bg_color, cell.bg_hovered_color, cell.bg_color, 60, False))
            self.next_turn()
        return bool(path)
            
    
    def draw(self) -> pg.Surface:
        size = self.padding + 9*(self.cell_size + self.padding)
        image = pg.Surface((size, size)) 
        image.fill(Colors.black.value)
        self.sprite_group.update()
        self.sprite_group.draw(image)
        return image
    
    
    def cells2pixels(self, coords: tuple[int, int]) -> tuple[int, int]:
        x, y = coords
        return self.padding + x*(self.cell_size + self.padding), self.padding + y*(self.cell_size + self.padding)
    
    
    def clicked_cell(self, pos: tuple[int, int]) -> Cell | None:
        for cell in self.all_cells():
            if cell.rect.collidepoint(pos): # type: ignore
                return cell
    
    
    def check_field(self) -> int:
        eliminated: set[Cell] = set()
        by_columns = [[self.field[y][x] for y in range(9)] for x in range(9)]
        by_right_diag = [[self.field[y+x][x] for x in range(9) if 0 <= x < 9 and 0 <= y+x < 9] for y in range(-4, 5)]
        by_left_diag = [[self.field[y-x][x] for x in range(9) if 0 <= x < 9 and 0 <= y-x < 9] for y in range(4, 13)]
        
        for view in (self.field, by_columns, by_right_diag, by_left_diag):
            for row in view:
                cur = None
                streak = 1
                before = len(eliminated)
                for i, cell in enumerate(row, start=1):
                    if cell.stage is not CellStages.occupied:
                        cur = None
                        streak = 0
                    elif cell.color is cur:
                        streak += 1
                        if streak >= 5:
                            eliminated |= set(row[i-streak:i])
                            # print(f'streak: {streak}')
                            # print([cell.coords for cell in row[i-streak:i]])
                            # print([cell.color.name for cell in row[i-streak:i]])
                            # print([cell.stage.name for cell in row[i-streak:i]])
                            # print('-------')
                    else:
                        cur = cell.color
                        streak = 1
                after = len(eliminated)
                if after-before >= 5:
                    self.stats[f'{after-before}_in_line'] += 1
                                 

        for cell in eliminated:
            cell.stage = CellStages.empty
            cell.fg_effects.clear()
            cell.bg_effects.clear()
            cell.bg_effects.append(ColorEffect(cell.bg_color, cell.color.value, cell.bg_color, 20, False)) # type: ignore
            self.stats[f'{cell.color.name}_popped'] += 1 # type: ignore
            cell.color = None
            
        return len(eliminated)
    
    
    def grow(self):
        for cell in self.seed_cells():
            cell.grow()
    
    
    def track_highscore(self):
        if self.score > self.stats["highscore"]:
            new_stats = {key : value for key, value in self.stats.items()}
            new_stats["highscore"] = self.score
            with open('data/stats.json', 'w') as f:
                json.dump(new_stats, f, indent=4)
    
    
    def next_turn(self):
        score = self.score
        self.score += self.check_field()
        self.track_highscore()
        
        if self.score == score:
            self.grow()
            self.score += self.check_field()
            self.track_highscore()
        
        self.spawn()
        self.turns += 1
    
    
FPS = 60
pg.init()
Textures.load()
game = Game()
w, h = game.draw().get_size()
screen = pg.display.set_mode((w, h + game.cell_size + game.padding))
manager = gui.UIManager(screen.get_size(), 'data/theme.json')
score_button = gui.elements.UIButton(relative_rect=pg.Rect((game.padding, h), (game.cell_size*3 + game.padding*2, game.cell_size)),
                                             text=f'Score: 0/{game.stats["highscore"]}',
                                             manager=manager,
                                             object_id=gui.core.ObjectID(object_id='0', class_id='@display'))
restart_button = gui.elements.UIButton(relative_rect=pg.Rect(((game.cell_size + game.padding)*3 + game.padding, h), (game.cell_size*3 + game.padding*2, game.cell_size)),
                                             text='Restart',
                                             manager=manager)
turns_button = gui.elements.UIButton(relative_rect=pg.Rect(((game.cell_size + game.padding)*6 + game.padding, h), (game.cell_size*3 + game.padding*2, game.cell_size)),
                                             text='Turns: 0',
                                             manager=manager,
                                             object_id=gui.core.ObjectID(object_id='1', class_id='@display'))
pg.display.set_icon(pg.image.load('data/icon.ico'))
clock = pg.time.Clock()

running = True
while running:
    for event in pg.event.get():
        manager.process_events(event)
        match event.type:
            case pg.QUIT:
                game.save_stats()
                running = False
                
            case pg.MOUSEBUTTONUP:
                target = game.clicked_cell(event.pos)
                if target is None:
                    game.select(None)
                    
                elif game.chosen is None and target.stage is CellStages.occupied:
                    game.select(target)
                    
                elif game.chosen is not None and target is not game.chosen:
                    if target.stage is CellStages.occupied:
                        game.select(target)
                    elif game.player_action(game.chosen, target):
                        game.chosen.deselect()
                        game.chosen = None
                    
            case pg.MOUSEMOTION:
                target = game.clicked_cell(event.pos)
                for cell in game.all_cells():
                    cell.hovered = False
                if target is not None:
                    target.hovered = True
                    
            case gui.UI_BUTTON_PRESSED:
                if event.ui_element == restart_button:
                    game = Game()
    
    if game.score > game.stats["highscore"]:
        pg.display.set_caption('Lines 98 - New Highscore!')
    score_button.set_text(f'Score: {game.score}/{game.stats["highscore"]}')
    turns_button.set_text(f'Turns: {game.turns}')
    manager.update(100/6)
    frame = game.draw()
    screen.blit(frame, (0, 0))
    manager.draw_ui(screen)
    clock.tick(FPS)
    pg.display.flip()
pg.quit()