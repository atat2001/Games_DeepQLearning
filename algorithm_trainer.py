from algorithm2 import Trainer, Step, Agent
from easier_hexagon import Game
from easier_hexagon import Player
from multiprocessing import Pool


# Constants
PROCESSES = 1  # used for threads but didnt work out 
SIZE = 3
WIDTH, HEIGHT = 1200, 1200
POINT_RADIUS = 20
FPS = 60
SQUARE = 2 * SIZE - 1
WIDTH_UNIT = (WIDTH/(SQUARE+2))
HEIGHT_UNIT = (HEIGHT/(SQUARE+2))



# Main function
def main():
    game = Game(SIZE, Player("player1"), Player("player2"))

    trainers = []
    actions = game.get_moves_idf()
    nr_actions = len(actions)
    input_shape = (19, 19)
    agents = [Agent(input_shape,input_shape,nr_actions), Agent(input_shape,input_shape,nr_actions)]
    # exit()
    trainer = Trainer(input_shape,agents, Game(SIZE, Player("player1"), Player("player2")))
    # points = [(100, 100), (200, 200), (300, 300)]  # Example button positions

    running = True

    curr_player = 0
    i = 0
    iterations = 1000
    while iterations != 0:            # connect(game,node1, node2, canvas)
        
        trainer.step()
        if trainer.game.check_game_finished():       
            iterations += -PROCESSES
    print("done")
if __name__ == "__main__":
    main()





