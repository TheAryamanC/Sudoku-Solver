import numpy as np
from typing import Optional, List


class SudokuSolver:
    
    def __init__(self):
        self.result: Optional[np.ndarray] = None
        self.call_counter: int = 0
        self.iterations_matrix: Optional[np.ndarray] = None
    
    def find_empty_cells(self, board: np.ndarray) -> np.ndarray:
        empty_positions = np.argwhere(board == 0)
        return empty_positions
    
    def is_valid(self, board: np.ndarray, num: int, row: int, col: int) -> bool:
        if num in board[row, :]:
            return False
        
        if num in board[:, col]:
            return False
        
        box_x = (row // 3) * 3
        box_y = (col // 3) * 3
        box = board[box_x:box_x + 3, box_y:box_y + 3]
        
        if num in box:
            return False
        
        return True
    
    def solve(self, board: np.ndarray, track_iterations: bool = False) -> bool:
        self.result = None
        self.call_counter = 0
        if track_iterations:
            self.iterations_matrix = np.zeros((9, 9), dtype=int)
        
        needed_cells = self.find_empty_cells(board)
        
        if len(needed_cells) == 0:
            self.result = board.copy()
            return True
        
        needed_cells = self._sort_cells_by_box_density(needed_cells)
        
        return self._solve_recursive(board.copy(), needed_cells, 0, track_iterations)
    
    def _sort_cells_by_box_density(self, needed_cells: np.ndarray) -> np.ndarray:
        if len(needed_cells) == 0:
            return needed_cells
        
        box_indices = []
        for row, col in needed_cells:
            box_idx = 3 * (row // 3) + (col // 3)
            box_indices.append(box_idx)
        
        box_counts = {}
        for box_idx in box_indices:
            box_counts[box_idx] = box_counts.get(box_idx, 0) + 1
        
        cell_priorities = [(box_counts[box_indices[i]], i) for i in range(len(needed_cells))]
        cell_priorities.sort(key=lambda x: x[0])
        
        sorted_indices = [x[1] for x in cell_priorities]
        return needed_cells[sorted_indices]
    
    def _solve_recursive(self, board: np.ndarray, needed_cells: np.ndarray, 
                         index: int, track_iterations: bool) -> bool:
        if index >= len(needed_cells):
            self.result = board.copy()
            return True
        
        row, col = needed_cells[index]
        
        for num in range(1, 10):
            if not self.is_valid(board, num, row, col):
                continue
            
            board[row, col] = num
            self.call_counter += 1
            
            if track_iterations:
                self.iterations_matrix[row, col] += 1
            
            if self._solve_recursive(board, needed_cells, index + 1, track_iterations):
                return True
            
            board[row, col] = 0
        
        return False
    
    def get_solution(self) -> Optional[np.ndarray]:
        return self.result
    
    def get_call_count(self) -> int:
        return self.call_counter
    
    def get_iterations_matrix(self) -> Optional[np.ndarray]:
        return self.iterations_matrix


def parse_board_string(board_string: str) -> np.ndarray:
    cleaned = board_string.replace('\n', '').replace(' ', '').replace(',', '')
    cleaned = cleaned.replace('.', '0').replace('_', '0')
    
    digits = [int(c) for c in cleaned if c.isdigit()]
    
    if len(digits) != 81:
        raise ValueError(f"Expected 81 digits, got {len(digits)}")
    
    return np.array(digits).reshape(9, 9)


def create_sample_board() -> np.ndarray:
    board = np.array([
        [0, 0, 0, 0, 3, 0, 0, 0, 1],
        [9, 8, 0, 0, 0, 0, 6, 0, 0],
        [0, 0, 0, 8, 0, 7, 0, 0, 0],
        [0, 0, 6, 2, 0, 4, 0, 7, 0],
        [0, 4, 0, 0, 0, 0, 9, 6, 0],
        [0, 7, 0, 5, 0, 3, 4, 8, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 7, 0, 0, 0, 0, 5, 9],
        [5, 0, 0, 0, 7, 0, 0, 0, 0]
    ])
    return board


if __name__ == "__main__":
    solver = SudokuSolver()
    board = create_sample_board()
    
    print("Original board:")
    print(board)
    print()
    
    if solver.solve(board, track_iterations=True):
        print("Solved board:")
        print(solver.get_solution())
        print(f"\nRecursive calls: {solver.get_call_count()}")
        print("\nIterations per cell:")
        print(solver.get_iterations_matrix())
    else:
        print("No solution found!")
