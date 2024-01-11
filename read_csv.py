import csv


def display_tic_tac_toe_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            state, action, value = row
            print("State before win:\n" + format_board(state))
            print("Last action:\n" + format_board(action))
            print("Reward:", value)
            print("\n" + "-" * 30 + "\n")


def format_board(board_str):
    formatted_board = ""
    for i in range(0, 9, 3):
        formatted_board += ' ' + ' | '.join(board_str[i:i + 3]) + '\n'
        if i < 6:
            formatted_board += "---+---+---\n"
    return formatted_board


# Example usage
display_tic_tac_toe_from_csv('q_table.csv')
