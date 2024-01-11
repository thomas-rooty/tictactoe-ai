import csv
import random

# Classe pour le jeu Tic Tac Toe
class TicTacToeGame:
    def __init__(self):
        self.state = '         '  # État initial du plateau de jeu
        self.player = 'X'  # Joueur actuel (X ou O)
        self.winner = None  # Gagnant du jeu

    # Retourne les mouvements possibles dans l'état actuel
    def allowed_moves(self):
        states = []
        for i in range(len(self.state)):
            if self.state[i] == ' ':
                states.append(self.state[:i] + self.player + self.state[i + 1:])
        return states

    # Effectue un mouvement et met à jour l'état du jeu
    def make_move(self, next_state):
        if self.winner:
            raise (Exception("Game already completed, cannot make another move!"))
        if not self.__valid_move(next_state):
            raise (Exception("Cannot make move {} to {} for player {}".format(
                self.state, next_state, self.player)))

        self.state = next_state
        self.winner = self.predict_winner(self.state)
        if self.winner:
            self.player = None
        elif self.player == 'X':
            self.player = 'O'
        else:
            self.player = 'X'

    # Vérifie si le jeu est encore jouable
    def playable(self):
        return (not self.winner) and any(self.allowed_moves())

    # Prédit le gagnant du jeu
    def predict_winner(self, state):
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        winner = None
        for line in lines:
            line_state = state[line[0]] + state[line[1]] + state[line[2]]
            if line_state == 'XXX':
                winner = 'X'
            elif line_state == 'OOO':
                winner = 'O'
        return winner

    # Vérifie si le mouvement est valide
    def __valid_move(self, next_state):
        allowed_moves = self.allowed_moves()
        if any(state == next_state for state in allowed_moves):
            return True
        return False

    # Affiche le plateau de jeu
    def print_board(self):
        s = self.state
        print('     {} | {} | {} '.format(s[0], s[1], s[2]))
        print('    -----------')
        print('     {} | {} | {} '.format(s[3], s[4], s[5]))
        print('    -----------')
        print('     {} | {} | {} '.format(s[6], s[7], s[8]))

# Classe pour l'agent d'apprentissage par renforcement Q-Learning
class QLearningAgent:
    def __init__(self, game_class, epsilon=0.1, alpha=0.5, gamma=0.9, value_player='X'):
        self.Q = dict()  # Table Q pour stocker les valeurs
        self.NewGame = game_class  # Classe de jeu
        self.epsilon = epsilon  # Taux d'exploration
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de remise
        self.value_player = value_player  # Joueur pour lequel on calcule la valeur

    # Apprend à jouer au jeu sur un certain nombre d'épisodes
    def learn_game(self, num_episodes=1000):
        for episode in range(num_episodes):
            self.learn_from_episode()

    # Apprend à partir d'un seul épisode
    def learn_from_episode(self):
        game = self.NewGame()
        state = game.state
        while game.playable():
            action = self.select_action(state, game.allowed_moves())
            reward, next_state = self.make_move_and_get_reward(game, action)
            self.update_q_table(state, action, reward, next_state)
            state = next_state

    # Sélectionne une action en fonction de la politique epsilon-greedy
    def select_action(self, state, allowed_moves):
        if random.random() < self.epsilon:
            return random.choice(allowed_moves)
        else:
            q_values = [self.get_q_value(state, action) for action in allowed_moves]
            max_q = max(q_values)
            max_q_actions = [action for action, q in zip(allowed_moves, q_values) if q == max_q]
            return random.choice(max_q_actions)

    # Effectue un mouvement et retourne la récompense
    def make_move_and_get_reward(self, game, action):
        game.make_move(action)
        reward = self.__reward(game)
        return reward, game.state

    # Met à jour la table Q
    def update_q_table(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in self.NewGame().allowed_moves()], default=0)
        self.Q[(state, action)] = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)

    # Obtient la valeur Q pour un état et une action donnés
    def get_q_value(self, state, action):
        return self.Q.get((state, action), 0.0)

    # Calcule la récompense
    def __reward(self, game):
        if game.winner == self.value_player:
            return 1.0
        elif game.winner:
            return -1.0
        else:
            return 0.0

    # Sauvegarde la table Q dans un fichier CSV
    def save_q_table(self):
        with open('q_table.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['State before win', 'Win action', 'Reward'])
            for (state, action), value in self.Q.items():
                writer.writerow([state, action, value])

    # Démontre un jeu en utilisant la meilleure stratégie
    def demo_game(self, verbose=False):
        game = self.NewGame()
        t = 0
        while game.playable():
            if verbose:
                print(" \nTurn {}\n".format(t))
                game.print_board()
            move = self.select_best_move(game)
            game.make_move(move)
            t += 1
        if verbose:
            print(" \nTurn {}\n".format(t))
            game.print_board()
        if game.winner:
            if verbose:
                print("\n{} is the winner!".format(game.winner))
            return game.winner
        else:
            if verbose:
                print("\nIt's a draw!")
            return '-'

    # Sélectionne le meilleur mouvement en fonction de la table Q
    def select_best_move(self, game):
        state = game.state
        allowed_moves = game.allowed_moves()
        q_values = [self.get_q_value(state, action) for action in allowed_moves]
        max_q = max(q_values)
        best_actions = [action for action, q in zip(allowed_moves, q_values) if q == max_q]
        return random.choice(best_actions)

# Fonction pour afficher les statistiques d'un jeu de démonstration
def demo_game_stats(agent):
    results = [agent.demo_game() for i in range(10000)]
    game_stats = {k: results.count(k) / 100 for k in ['X', 'O', '-']}
    print("    percentage results: {}".format(game_stats))

# Point d'entrée principal
if __name__ == '__main__':
    agent = QLearningAgent(TicTacToeGame, epsilon=0.1, alpha=1.0, gamma=0.9)
    print("Before learning:")
    demo_game_stats(agent)

    agent.learn_game(1000)
    print("After 1000 learning games:")
    demo_game_stats(agent)

    agent.learn_game(4000)
    print("After 5000 learning games:")
    demo_game_stats(agent)

    agent.learn_game(5000)
    print("After 10000 learning games:")
    demo_game_stats(agent)

    agent.learn_game(10000)
    print("After 20000 learning games:")
    demo_game_stats(agent)

    agent.learn_game(10000)
    print("After 30000 learning games:")
    demo_game_stats(agent)

    agent.save_q_table()
