import sys
import numpy as np
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent))

from solver import SudokuSolver, parse_board_string, create_sample_board
from visualizer import print_board, print_solution_comparison, display_board_matplotlib


def get_manual_input() -> np.ndarray:
    print("\n" + "=" * 60)
    print("MANUAL SUDOKU INPUT")
    print("=" * 60)
    print("\nEnter the sudoku puzzle in one of these formats:")
    print("  1. 81 digits in a row (use 0 or . for empty cells)")
    print("  2. 9 lines of 9 digits each")
    print("  3. Type 'sample' to use a sample puzzle")
    print("\nExample: 530070000600195000098000060800060003400803001700020006060000280000419005000080079")
    print()
    
    lines = []
    print("Enter puzzle (press Enter twice when done, or enter 81 digits on one line):")
    
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        
        if line.lower() == 'sample':
            print("\nUsing sample puzzle...")
            return create_sample_board()
        
        if not line:
            if lines:
                break
            continue
        
        lines.append(line)
        
        combined = ''.join(lines)
        digits = [c for c in combined if c.isdigit() or c in '.']
        if len(digits) >= 81:
            break
    
    if not lines:
        raise ValueError("No input provided")
    
    board_string = ''.join(lines)
    return parse_board_string(board_string)


def get_image_input() -> np.ndarray:
    print("\n" + "=" * 60)
    print("IMAGE INPUT")
    print("=" * 60)
    
    try:
        from image_reader import SudokuImageReader
        reader = SudokuImageReader()
        reader._ensure_dependencies()
    except ImportError as e:
        print("\nTo use image input, install the required packages:")
        print("  pip install opencv-python tensorflow")
        print()
        raise
    
    print("\nEnter the path to your sudoku image:")
    print("(Supports: .jpg, .jpeg, .png, .bmp)")
    
    image_path = input("\nImage path: ").strip()
    
    image_path = image_path.strip('"\'')
    
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    board = reader.read_sudoku_from_image(image_path)
    
    print("\n✓ Successfully extracted puzzle from image!")
    return board


def solve_and_display(board: np.ndarray, use_matplotlib: bool = False) -> None:
    print("\n" + "=" * 60)
    print("SOLVING PUZZLE")
    print("=" * 60)
    
    print_board(board, "Original Puzzle")
    
    solver = SudokuSolver()
    
    print("\nSolving...", end=" ", flush=True)
    
    if solver.solve(board.copy(), track_iterations=True):
        print("✓ Solved!\n")
        
        solution = solver.get_solution()
        
        print_solution_comparison(board, solution, "Solution")
        
        print(f"\n📊 Statistics:")
        print(f"   Recursive calls: {solver.get_call_count()}")
        
        if use_matplotlib:
            display_board_matplotlib(solution, board, "Sudoku Solution")
        else:
            try:
                show_plot = input("\nShow graphical display? (y/n): ").strip().lower()
                if show_plot == 'y':
                    display_board_matplotlib(solution, board, "Sudoku Solution")
            except:
                pass
    else:
        print("✗ No solution found!")
        print("\nThe puzzle may be invalid. Please check your input.")


def validate_board(board: np.ndarray) -> Tuple[bool, str]:
    if board.shape != (9, 9):
        return False, f"Invalid dimensions: {board.shape}, expected (9, 9)"
    
    if np.any(board < 0) or np.any(board > 9):
        return False, "Values must be between 0 and 9"
    
    for i in range(9):
        row = board[i, :]
        non_zero = row[row > 0]
        if len(non_zero) != len(set(non_zero)):
            return False, f"Duplicate numbers in row {i + 1}"
    
    for j in range(9):
        col = board[:, j]
        non_zero = col[col > 0]
        if len(non_zero) != len(set(non_zero)):
            return False, f"Duplicate numbers in column {j + 1}"
    
    for box_row in range(3):
        for box_col in range(3):
            box = board[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
            non_zero = box[box > 0].flatten()
            if len(non_zero) != len(set(non_zero)):
                return False, f"Duplicate numbers in box ({box_row + 1}, {box_col + 1})"
    
    return True, ""


def main():
    print("\n" + "=" * 60)
    print("       🧩 SUDOKU SOLVER 🧩")
    print("=" * 60)
    print("\nChoose input method:")
    print("  1. Manual input (type the numbers)")
    print("  2. Import from image (uses ML)")
    print("  3. Use sample puzzle")
    print("  4. Exit")
    
    while True:
        try:
            choice = input("\nYour choice (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            return
        
        if choice == '1':
            try:
                board = get_manual_input()
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
                
        elif choice == '2':
            try:
                board = get_image_input()
            except FileNotFoundError as e:
                print(f"\n❌ Error: {e}")
                continue
            except ImportError:
                continue
            except Exception as e:
                print(f"\n❌ Error processing image: {e}")
                continue
                
        elif choice == '3':
            board = create_sample_board()
            print("\nUsing sample puzzle...")
            
        elif choice == '4':
            print("\nGoodbye!")
            return
            
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
            continue
        
        is_valid, error_msg = validate_board(board)
        if not is_valid:
            print(f"\n❌ Invalid puzzle: {error_msg}")
            continue
        
        solve_and_display(board)
        
        try:
            again = input("\nSolve another puzzle? (y/n): ").strip().lower()
            if again != 'y':
                print("\nGoodbye!")
                return
        except:
            return
        
        print("\n" + "=" * 60)
        print("       🧩 SUDOKU SOLVER 🧩")
        print("=" * 60)
        print("\nChoose input method:")
        print("  1. Manual input (type the numbers)")
        print("  2. Import from image (uses ML)")
        print("  3. Use sample puzzle")
        print("  4. Exit")


if __name__ == "__main__":
    main()
