from algorithm2 import Trainer, Step, Agent
import pygame
import sys
import time
import numpy as np
from easier_hexagon import Game
from easier_hexagon import Player
import os

# Set the position of the display window
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 100)


def get_action_index(game,node1, node2, canvas, actions):
    idf1 = node1.unique_idf
    idf2 = node2.unique_idf
    for action_index in range(len(actions)):
        action = actions[action_index]
        if(action[0] == idf1):
            if(action[1] == idf2):
                return action_index
        if(action[0] == idf2):
            if(action[1] == idf1):
                return action_index
    return None

class Me_as_agent:
    # def __init__(self,input_shape1, input_shape2,num_actions):
    def __init__(self, state_shape1, state_shape2, num_actions,canvas, trainer, actions, learning_rate=0.001, discount_factor=0.99):
        self.state_shape1 = state_shape1
        self.state_shape2 = state_shape2
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.agent_index = 1
        self.actions = actions
        # Define the Q-network
        self.q_network = self._build_q_network()
        self.human_input = True
        self.canvas = canvas
        self.trainer = trainer
        # Compile the Q-network

        self.load_weights()

    def _build_q_network(self):
        return


    def select_actions(self, state):
        canvas = self.canvas
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        if canvas.selected != 0:
                            canvas.last_selected = canvas.selected
                            canvas.selected = 0
                        # Check if any point (button) is clicked
                        for point in canvas.points:
                            if is_point_clicked(point, event.pos):
                                canvas.selected = point[2]
                                # print(point[2].to_string())

                        if canvas.selected != 0 and canvas.last_selected != 0:
                            # players_turn.play(game, selected, last_selected, lines)
                            action_index = get_action_index(self.trainer.game,canvas.selected, canvas.last_selected, canvas, self.actions)
                            var = np.zeros(self.num_actions)
                            var[action_index] = 101
                            canvas.selected = 0
                            canvas.last_selected = 0
                            return var
                        elif canvas.selected == 0:
                            canvas.last_selected = 0
        return var
    # step list -> batch
    def train(self, batch):
        return

    def load_weights(self):
        return

    def save_weights(self):
        return
    
# to debug ml
DRAW_NUMBERS = True



# Initialize Pygame
pygame.init()
pygame.font.init()

# Constants
SIZE = 3
WIDTH, HEIGHT = 1200, 1200
POINT_RADIUS = 20
FPS = 60
SQUARE = 2 * SIZE - 1
WIDTH_UNIT = (WIDTH/(SQUARE+2))
HEIGHT_UNIT = (HEIGHT/(SQUARE+2))

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0,0,255)

# Function to check if a point (button) is clicked
def is_point_clicked(position, click):
    return (position[0] - click[0]) ** 2 + (position[1] - click[1]) ** 2 <= POINT_RADIUS ** 2

def convert_x_to_hexagon(node_x, node_y):
    nr_spaces = SIZE - 1
    return (node_x * WIDTH_UNIT) + WIDTH_UNIT + ((WIDTH_UNIT/2) * (nr_spaces-node_y))

def convert_y_to_hexagon(node_x, node_y):
    return node_y * WIDTH_UNIT + WIDTH_UNIT

def get_points(game):
    points = []
    for node_list in game.get_nodes():
        for node in node_list:
            points = points + [(convert_x_to_hexagon(node.x, node.y), convert_y_to_hexagon(node.x, node.y), node),]
            pos1 = [node.x,node.y]
    return points

def connect(game,node1, node2, canvas):
    var = game.connect(node1, node2)
    input_matrix_shape = (18,3)
    # print(game.get_moves_idf())
    if var != -1:
        canvas.lines.append([[convert_x_to_hexagon(node1.x,node1.y),convert_y_to_hexagon(node1.x,node1.y)], [convert_x_to_hexagon(node2.x,node2.y),convert_y_to_hexagon(node2.x,node2.y)]])
    return 1

def convert_triangles_to_positions(triangles_list):
    triangles = []
    for triangle in triangles_list:
        if(triangle == []):
            continue
        triangle_vertices = []
        for node in triangle:
            triangle_vertices = triangle_vertices + [(convert_x_to_hexagon(node.x, node.y), convert_y_to_hexagon(node.x, node.y)),]
        triangles = triangles + [triangle_vertices,]
    return triangles

def get_triangles(game):
    triangles = []
    for player in game.players:
        for triangle in convert_triangles_to_positions(player.get_triangles()):
            triangles = triangles + triangle
    return triangles

class Canvas:
    def __init__(self,surface,points, selected, last_selected, lines, triangles, font, clock, game):
        self.surface = surface
        self.points = points
        self.selected = selected
        self.last_selected = last_selected
        self.lines = []
        self.triangles = []
        self.font = font
        self.game = game
        self.clock = clock
        
    # Function to draw a point (button)
    def draw_point(self,color, position, font):
        pygame.draw.circle(self.surface, color, position[0:2], POINT_RADIUS)
        if(DRAW_NUMBERS):
            # Render the number
            stringe = str(position[2].unique_idf) + "\n" + str(position[2].x) + ", " + str(position[2].y)
            text_surface = font.render(stringe, True, BLUE)
            text_rect = text_surface.get_rect(center=position[0:2])
            self.surface.blit(text_surface, text_rect)

    # Function to draw a line
    def draw_line(self,color, positions):
        # Draw a line from pos1 to pos2 with RED color and width 5
        pygame.draw.line(self.surface, RED, positions[0], positions[1], 5)

    # Function to draw a triangle
    def paint_triangle(self, color, positions):
        pygame.draw.polygon(self.surface, color, positions)


    def end_screen(self):
        # Clear the screen
        screen = self.surface
        screen.fill(WHITE)

        # Draw points (buttons)
        for point in self.points:
            if(point[2] == self.selected or point[2] == self.last_selected):
                self.draw_point(screen, BLUE, point, self.font)
            else:
                self.draw_point(screen, BLACK, point, self.font)
        for line in self.lines:
            self.draw_line(screen, RED, line)
        score_y = 10
        for player in self.game.players:
            for triangle in convert_triangles_to_positions(player.get_triangles()):
                self.paint_triangle(screen, player.color, triangle)
            score_text = self.font.render(f"{player.name}: {player.points}", True, BLACK)
            screen.blit(score_text, (10, score_y))
            score_y += 40
        # Update the display
        pygame.display.flip()
        # Cap the frame rate
        self.clock.tick(FPS)
        time.sleep(3)
        pygame.quit()
        sys.exit()

    def draw_points(self):
        # Draw points (buttons)
        for point in self.points:
            if(point[2] == self.selected or point[2] == self.last_selected):
                self.draw_point(BLUE, point,self.font)
            else:
                self.draw_point(BLACK, point,self.font)

    def draw_lines(self):
        for line in self.lines:
            self.draw_line(RED, line)

    def draw_triangles(self):
        for player in self.game.players:
            for triangle in convert_triangles_to_positions(player.get_triangles()):
                 self.paint_triangle(player.color, triangle)
        return 
           
        
    def draw_score(self):
        score_y = 10
        for player in self.game.players:
            score_text = self.font.render(f"{player.name}: {player.points}", True, BLACK)
            self.surface.blit(score_text, (10, score_y))
            score_y += 40
    
    def draw(self):
        # Clear the screen
        self.surface.fill(WHITE)
        self.draw_points()
        self.draw_lines()
        self.draw_triangles()
        self.draw_score()
        # Update the display
        pygame.display.flip()
        # Cap the frame rate
        self.clock.tick(FPS)

# Main function
def main():
    x_position = 100  # Adjust as needed
    y_position = 100  # Adjust as needed
    game = Game(SIZE, Player("player1"), Player("player2"))


    actions = game.get_moves_idf()
    nr_actions = len(actions)
    input_shape = (19, 19)
    agents = [Agent(input_shape,input_shape,nr_actions), Agent(input_shape,input_shape,nr_actions)]
    #(self, state_shape1, state_shape2, num_actions,canvas, trainer, actions, learning_rate=0.001, discount_factor=0.99) 
    trainer = Trainer(input_shape,agents, game)
    x_position += -100
    y_position += -100
    # Set the position of the display window
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x_position, y_position)
    # Set up the display
    
    canvas = Canvas(pygame.display.set_mode((WIDTH, HEIGHT)),get_points(game), 0, 0, [], [], pygame.font.Font(None, 36), pygame.time.Clock(), game)
    pygame.display.set_caption("Point UI")
    trainer.agents[1] = Me_as_agent(input_shape,input_shape,nr_actions,canvas,trainer,actions)



    # points = [(100, 100), (200, 200), (300, 300)]  # Example button positions

    running = True

    curr_player = 0
    i = 0
    iterations = 1000
    while iterations != 0:            # connect(game,node1, node2, canvas)
        #print("before step")
        trainer.step()
        #print("after step")
        step = trainer.get_last_step()    # game.make_action(self.actions[move_selected])
        #actions[step.move_selected]       # self.connect(self.nodes_idf_dict[action[0]],self.nodes_idf_dict[action[1]])
        connect(game,game.nodes_idf_dict[actions[step.move_selected][0]],game.nodes_idf_dict[actions[step.move_selected][1]], canvas)
        #print("connecteded")
        # print(f"here{i}\n")
        i += 1
        canvas.draw()
        if trainer.game.check_game_finished():       
            time.sleep(0.1)

            trainer.step()
            iterations += -1
            canvas = Canvas(canvas.surface,get_points(game), 0, 0, [], [], pygame.font.Font(None, 36), pygame.time.Clock(), trainer.game)

      
    pygame.quit()
    sys.exit()
if __name__ == "__main__":
    main()





