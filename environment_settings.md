# Environment Settings

Definitions live in `LAYOUTS` in `environment.py`. Coordinates are `(row, col)`, with `(0, 0)` at the top-left.

Map legend: `#` obstacle, `S` candidate start, `G` goal, `.` free cell.

## Shared rules

| Setting | Value |
|---|---|
| Actions | `0` up, `1` right, `2` down, `3` left |
| Observation | `(row, col) / (size - 1)`, shape `(2,)`, range 0–1 |
| Step reward | -0.1 |
| Goal reward | +10, episode terminates |
| Collision reward | -10, episode terminates, position unchanged |
| Episode limit | 100 steps by default (`max_steps`), then truncated |

A collision is any move into a wall or obstacle.

## Layouts

Each environment instance uses one layout for its whole lifetime. The observation does not encode the map, so layouts should not be switched between episodes without changing the observation.

"Shortest path" is the number of steps from that start to the goal. Every free cell in every layout can reach the goal.

### `default_6x6`

The default for `GridWorld(size=6)`. Goal `(5, 5)`, 6 obstacles.

```
S.....
.##...
....#.
.#.#..
...#..
.....G
```

| Start | Shortest path |
|---|---|
| (0, 0) | 10 |

Obstacles: (1,1), (1,2), (2,4), (3,1), (3,3), (4,3)

### `rooms`

The default for `GridWorld(size=10)`. Four rooms joined by doorways. Goal `(9, 9)`, 15 obstacles.

```
S...#....S
....#.....
..........
....#.....
##.###.###
....#.....
S...#.....
..........
....#.....
....#....G
```

| Start | Shortest path |
|---|---|
| (0, 0) | 18 |
| (0, 9) | 15 |
| (6, 0) | 12 |

Obstacles: (0,4), (1,4), (3,4), (4,0), (4,1), (4,3), (4,4), (4,5), (4,7), (4,8), (4,9), (5,4), (6,4), (8,4), (9,4)

### `zigzag`

Long walls force a back-and-forth route. Goal `(9, 9)`, 28 obstacles.

```
S.........
#######...
..........
...#######
.........S
#######...
..........
...#######
S.........
.........G
```

| Start | Shortest path |
|---|---|
| (0, 0) | 38 |
| (4, 9) | 19 |
| (8, 0) | 10 |

Obstacles: (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (7,3), (7,4), (7,5), (7,6), (7,7), (7,8), (7,9)

### `clutter`

Scattered obstacles with many short detours. Goal `(9, 9)`, 28 obstacles.

```
S..#......
.#...##.#.
.#.#....#.
...#.##...
.#....#.#S
.##.#...#.
....#.#...
S#.....##.
.#.##.#...
...#.....G
```

| Start | Shortest path |
|---|---|
| (0, 0) | 18 |
| (4, 9) | 5 |
| (7, 0) | 13 |

Obstacles: (0,3), (1,1), (1,5), (1,6), (1,8), (2,1), (2,3), (2,8), (3,3), (3,5), (3,6), (4,1), (4,6), (4,8), (5,1), (5,2), (5,4), (5,8), (6,4), (6,6), (7,1), (7,7), (7,8), (8,1), (8,3), (8,4), (8,6), (9,3)

## Selecting a configuration

```python
GridWorld(size=6)                       # default_6x6
GridWorld(size=10)                      # rooms
GridWorld(layout='zigzag')              # named layout; sets size, obstacles, starts, goal
GridWorld(layout='zigzag', start=(0, 0))        # one fixed start
GridWorld(layout='zigzag', starts=[(0, 0), (8, 0)])  # custom candidate starts
GridWorld(size=8, obstacles={(1, 1)})   # custom map; start (0, 0), goal in the far corner
```

`layout` cannot be combined with `obstacles` or `goal`. `start` and `starts` cannot be combined.

## Start and goal rules

A start must be inside the grid, not on an obstacle, not the goal, and able to reach the goal. A goal must be inside the grid and not on an obstacle. Invalid configurations raise `ValueError`.

## Seeding

- `reset(seed=s)` always selects the same start for the same `s`, with or without Gymnasium installed.
- `reset()` with no seed continues the current random stream.
- With one candidate start, every reset uses that start.
- `reset(options={'start': (r, c)})` forces a validated start. Use it for common evaluation scenarios across methods.
- `reset` returns `info = {'start': (r, c), 'layout': name}`. `name` is `None` for custom maps.

## Shield

`shield.choose_action` uses `GridWorld.safe_actions()`, the same wall and obstacle rules as `step`.

- An unsafe proposal is replaced by a safe action according to the method; `unshielded` never intervenes.
- A safe proposal is executed unchanged.
- If a cell has no safe action, the proposal is executed and reported as `no_safe_action`. No candidate start in the named layouts is enclosed like this.
