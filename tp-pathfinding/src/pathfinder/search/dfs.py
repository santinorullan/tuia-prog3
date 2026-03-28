from ..models.grid import Grid
from ..models.frontier import StackFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node


class DepthFirstSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        # Initialize root node
        root = Node("", state=grid.initial, cost=0, parent=None, action=None)

        # Initialize expanded (visited)
        expanded = dict()
        expanded[root.state] = True

        # Initialize frontier with the root node (STACK → DFS)
        frontier = StackFrontier()
        frontier.add(root)

        # DFS loop
        while not frontier.is_empty():
            node = frontier.pop()

            # Goal test
            if grid.objective_test(node.state):
                return Solution(node, expanded)

            # Expand node
            for action in grid.actions(node.state):
                new_state = grid.result(node.state, action)

                if new_state not in expanded:
                    expanded[new_state] = True

                    child = Node(
                        "",
                        state=new_state,
                        cost=node.cost + grid.individual_cost(node.state, action),
                        parent=node,
                        action=action,
                    )

                    frontier.add(child)

        return NoSolution(expanded)
