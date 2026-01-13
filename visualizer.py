import numpy as np
from typing import Optional


def print_board(board: np.ndarray, title: str = "Sudoku Board") -> None:
    print(f"\n{title}")
    print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")
    
    for i in range(9):
        row_str = "|"
        for j in range(9):
            val = board[i, j]
            cell = str(val) if val != 0 else "."
            row_str += f" {cell}"
            if (j + 1) % 3 == 0:
                row_str += " |"
        print(row_str)
        if (i + 1) % 3 == 0:
            print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")


def print_solution_comparison(original: np.ndarray, solved: np.ndarray, 
                               title: str = "Sudoku Solution") -> None:
    print(f"\n{title}")
    print("(Original numbers in regular text, solved numbers marked with *)")
    print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")
    
    for i in range(9):
        row_str = "|"
        for j in range(9):
            if original[i, j] != 0:
                cell = str(original[i, j])
            elif solved[i, j] != 0:
                cell = str(solved[i, j])
            else:
                cell = "."
            row_str += f" {cell}"
            if (j + 1) % 3 == 0:
                row_str += " |"
        print(row_str)
        if (i + 1) % 3 == 0:
            print("+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+")


def board_to_string(board: np.ndarray) -> str:
    return ''.join(str(x) for x in board.flatten())


def display_board_matplotlib(board: np.ndarray, original: Optional[np.ndarray] = None,
                              title: str = "Sudoku") -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not installed. Using console display instead.")
        if original is not None:
            print_solution_comparison(original, board, title)
        else:
            print_board(board, title)
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    for i in range(9):
        for j in range(9):
            rect = patches.Rectangle((j, 8 - i), 1, 1, linewidth=1, 
                                       edgecolor='black', facecolor='white')
            ax.add_patch(rect)
            
            val = board[i, j]
            if val != 0:
                if original is not None and original[i, j] != 0:
                    color = 'black'
                    weight = 'bold'
                else:
                    color = 'blue'
                    weight = 'normal'
                
                ax.text(j + 0.5, 8 - i + 0.5, str(val), 
                       ha='center', va='center', fontsize=14,
                       color=color, fontweight=weight)
    
    for i in range(4):
        ax.axhline(y=i * 3, color='black', linewidth=3)
        ax.axvline(x=i * 3, color='black', linewidth=3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from solver import create_sample_board, SudokuSolver
    
    board = create_sample_board()
    print_board(board, "Test Board")
    
    solver = SudokuSolver()
    if solver.solve(board):
        print_solution_comparison(board, solver.get_solution(), "Solved")
        display_board_matplotlib(solver.get_solution(), board, "Sudoku Solved")
