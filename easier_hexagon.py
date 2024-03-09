import math
import random

class Player:
    def __init__(self, name):
        self.triangles = []
        self.name = name
        self.game = None
        self.color = random.choices(range(256), k=3)
        self.points = 0

    def add_triangles(self):
        self.triangles = self.game.get_player_triangles(self)
    
    def get_triangles(self):
        self.add_triangles()
        self.points = len(self.triangles)
        return self.triangles
    
    def add_game(self,game):
        self.game = game

    def play(self, node1, node2):
        return self.game.connect(node1, node2)

class Game:
    def __init__(self, size):
        self.nodes_idf_dict = dict()
        self.size = size
        self.move_order = []
        self.cur_move = 0
        self.graph = Graph(size)
        self.triangles = 0
        self.players = []
        self.moves = self.init_moves()
        self.moves_idf = self._init_moves_idf()

    def __init__(self, size, player1, player2):
        self.nodes_idf_dict = dict()
        self.size = size
        self.move_order = []
        self.cur_move = 0
        self.graph = Graph(size)
        self.triangles = 0
        self.players = []
        self.moves = self.init_moves()
        self.moves_idf = self._init_moves_idf()
        self.add_player(player1)
        self.add_player(player2)


    def init_moves(self):
        moves = [[],[],[]]
        vectors = [[0,1], [1,1], [1,0]]
        for nodes in self.graph.nodes:
            for node in nodes:
                for vector_index in [0,1,2]:
                    x = node.x + vectors[vector_index][0]
                    y = node.y + vectors[vector_index][1]
                    if self.graph.check_index(x,y):
                        moves[vector_index] = moves[vector_index] + [[[node.x,node.y], [x,y]]]
        return moves
    
    def _init_moves_idf(self):
        moves = []
        vectors = [[0,1], [1,1], [1,0]]
        for nodes in self.graph.nodes:
            for node in nodes:
                self.nodes_idf_dict[node.unique_idf] = node
                # vectors = [[-1,0], [0,1], [1,1], [1,0], [0,-1], [-1,-1]]
                if(node.adjacent_nodes[1] == 0):
                    if self.graph.check_index(node.x, node.y+1):
                        """print(self.graph.square_notation)"""
                        """print(f"add_move:{node.unique_idf},{self.get_node(node.x, node.y + 1).unique_idf}")
                        node.to_string()
                        self.get_node(node.x, node.y + 1).to_string()"""

                        moves = moves + [[node.unique_idf, self.get_node(node.x, node.y + 1).unique_idf],]
                if(node.adjacent_nodes[2] == 0):
                    if self.graph.check_index(node.x+1, node.y+1):
                        """
                        print(f"add_move:{node.unique_idf},{self.get_node(node.x+1, node.y + 1).unique_idf}")
                        node.to_string()
                        self.get_node(node.x+1, node.y + 1).to_string()"""

                        moves = moves + [[node.unique_idf, self.get_node(node.x+1, node.y + 1).unique_idf],]
                if(node.adjacent_nodes[3] == 0):
                    if self.graph.check_index(node.x+1, node.y):
                        """
                        print(f"add_move:{node.unique_idf},{self.get_node(node.x+1, node.y).unique_idf}")
                        node.to_string()
                        self.get_node(node.x+1, node.y).to_string()"""

                        moves = moves + [[node.unique_idf, self.get_node(node.x+1, node.y).unique_idf],]
        #print(len(moves))
        return moves
    
    def make_action(self, action):
        return self.connect(self.nodes_idf_dict[action[0]],self.nodes_idf_dict[action[1]])
    
    def get_moves_idf(self):
        return self.moves_idf

    ## 9*(size-1)*(size-1) + 4 * (size-1)
    def get_number_of_moves(self):
        return len(self.moves_idf)
        
    def add_player(self,player):
        self.players = self.players + [player,]
        player.add_game(self)

    def get_node(self, x, y):
        return self.graph.get_node(x,y)

    def connect(self, node1, node2):
        # self.remove_move_idf(node1, node2)
        # print(f"moves available: {self.get_number_of_moves()}")
        return self.graph.connect(node1, node2)

    def get_nodes(self):
        return self.graph.get_nodes()
    
    def get_player_triangles(self, player):
        index = self.players.index(player)
        nr_of_players = len(self.players)
        return self.graph.get_triangles_of_player(index, nr_of_players)

    def check_game_finished(self):
        return self.graph.check_game_finished()
    
    def get_point_winning_moves(self):
        return []


class Graph:
    def __init__(self, size):   
        self.idf_nodes = dict()
        self.size = size
        self.square_notation = 2 * size - 1
        self.nodes = self._init_nodes()
        self.triangles_history = dict()
        self.triangles_history[0] = []
        self.triangles = 0
        self.turn = 0

    def connect(self,node1,node2):
        if(node1 == node2):
            return -1
        delta_x = node1.x - node2.x
        delta_y = node1.y - node2.y
        distance = -1
        if delta_x == 0:
            distance = abs(delta_y)
            vector = [0,1]
            if delta_y < 0:
                vector = [0,-1]
        elif delta_y == 0:
            distance = abs(delta_x)
            vector = [1,0]
            if delta_x < 0:
                vector = [-1,0]
        elif delta_x == delta_y:
            distance = abs(delta_x)
            vector = [1,1]
            if delta_x < 0:
                vector = [-1,-1]
        else:
            print("Board->connect: nodes not inline")
            return -1
        if(distance > 1):
            return -1
        points_gained = 1
        turner = 0
        if(self.triangles_history.get(self.turn, None) == None):
            self.triangles_history[self.turn] = []
        for e in range(distance):
            if(self.add_edge(self.get_node(node2.x + e * vector[0], node2.y + e * vector[1]),self.get_node(node2.x + (e+1) * vector[0], node2.y + (e+1) * vector[1]))):
                before = len(self.triangles_history[self.turn])
                # points_gained += self.get_node(node2.x + (e+1) * vector[0], node2.y + (e+1) * vector[1]).count_edge_triangles(self.get_node(node2.x + e * vector[0], node2.y + e * vector[1]))
                self.triangles_history[self.turn] = self.triangles_history[self.turn] + self.get_node(node2.x + (e+1) * vector[0], node2.y + (e+1) * vector[1]).get_edge_triangles(self.get_node(node2.x + e * vector[0], node2.y + e * vector[1]))
                after = len(self.triangles_history[self.turn])
                if(after - before == 0):
                    # print("NO TRIANGLE")
                    points_gained = 0
                    turner = 1
                    self.triangles_history[self.turn + 1] = []
                else:
                    points_gained = after - before
                    self.triangles += after - before
        self.turn = self.turn + turner
        return points_gained


    ## since its an hexagon represented as a square we have to filter some squares
    ## this functions helps validate them
    def check_index(self, x,y):
        if not(x >= 0 and y >= 0 and x < self.square_notation and y < self.square_notation):
            return False
        middle = (self.square_notation - 1)/2
        diff = middle - x
        if diff > 0:
            if y > self.square_notation - diff - 1:
                return False
        if diff < 0:
            """
            if(x == 3):
                print(f"x:{x}, y:{y}, diff: {diff}")"""
            if y < -1*diff:
                """
                if(x == 3):
                    print("false")
                    """
                return False
        
        return True

    def get_node(self, x ,y):
        if not(self.check_index(x, y)):
            raise IndexError(f"Index {x}, {y} out of bounds")

        ## this code is necessary bc in the last few lines the elements i took out were the first ones
        ## due to the design, so we have to make the y values lower to match the right one on the list
        middle = (int)((self.square_notation - 1)/2)
        diff = middle - x
        if(diff < 0):
            y = y + diff

        return self.nodes[x][y]

    def get_nodes(self):
        return self.nodes
    
    def check_game_finished(self):
        if(6*(self.size-1)*(self.size-1) == self.triangles):
            return True
        return False

    def add_edge(self, node1, node2):
        if(node1.add_edge(node2)):
            node2.add_edge(node1)
            return True
        return False

    def _init_nodes(self):
        x = 0
        y = 0
        unique_idf = 0
        returner = []
        middle = (self.square_notation - 1)/2
        while x < self.square_notation:
            y = 0
            returner = returner + [[],]
            diff = middle - x
            if diff > 0:
                while y < self.square_notation:
                    if y > self.square_notation - diff - 1:
                        y = y + 1
                        continue
                    node = Node(x,y,self, unique_idf)
                    self.idf_nodes[unique_idf] = node
                    returner[x] = returner[x] + [node]
                    unique_idf += 1
                    y = y + 1
                
            elif diff < 0:
                while y < self.square_notation:
                    if y < -diff:
                        y = y + 1
                        continue
                    node = Node(x,y,self, unique_idf)
                    self.idf_nodes[unique_idf] = node
                    returner[x] = returner[x] + [node]
                    unique_idf += 1
                    y = y + 1
                
            else:
                while y < self.square_notation:
                    node = Node(x,y,self, unique_idf)
                    self.idf_nodes[unique_idf] = node
                    returner[x] = returner[x] + [node]
                    unique_idf += 1
                    y = y + 1
            x = x + 1
        return returner

    def get_triangles_of_player(self, index_of_player, nr_of_players):
        curr_index = index_of_player
        triangles = []
        while curr_index <= self.turn:
            triangles = triangles + self.triangles_history.get(curr_index, [])
            curr_index = curr_index + nr_of_players
        return triangles

    ## unnecessary
    def to_string(self):
        nr_spaces = self.square_notation - self.size
        diff = -1
        self.nodes.reverse()
        for x in self.nodes:
            if(nr_spaces == -1):
                diff = 1
                nr_spaces = 1
            z = 0
            while z < nr_spaces:
                print(" ", end="")
                z = z+1
            nr_spaces = nr_spaces + diff
            for y in x:
                print(f"[{y.x},{y.y}]", end="")
            print("\n", end="")
        self.nodes.reverse()
    
    ## innecessary: calculates numbeer of nodes
    def calculate_nodes(self):
        return ((self.size-1)*(self.size)*3) + 1
        

class Node:
    def __init__(self, x, y, graph):
        self.x = x
        self.y = y
        self.adjacent_nodes = [0,0,0,0,0,0]
        self.graph = graph
        self.nr_triangles = 0
        self.unique_idf = -1
    
    def __init__(self, x, y, graph, unique_idf):
        self.x = x
        self.y = y
        self.adjacent_nodes = [0,0,0,0,0,0]
        self.graph = graph
        self.nr_triangles = 0
        self.unique_idf = unique_idf
    
    def add_edge(self, node):
        vectors = [[-1,0], [0,1], [1,1], [1,0], [0,-1], [-1,-1]]
        try:
            index = self.get_adjacent_node_index(node)
            if self.adjacent_nodes[index] != 0:
                return False
            self.adjacent_nodes[index] = node
            return True
        except:
            print("node->get_adjacent_node_index: node is not adjacent")
            return False
    
    def get_adjacent_node_index(self, node):
        vector = [node.x - self.x, node.y - self.y]
        if(vector[0] == 0):
            index = 4
            if(vector[1] == 1):
                index = 1
        elif vector[0] == 1:
            index = 3
            if(vector[1] == 1):
                index = 2
        elif vector[0] == -1:
            index = 0
            if(vector[1] == -1):
                index = 5
        else:
            raise LookupError("node->get_adjacent_node_index: node is not adjacent")
        return index

    ## count the triangles in a specific edge
    def count_edge_triangles(self, node):

        nr_triangles = 0
        vectors = [[-1,0], [0,1], [1,1], [1,0], [0,-1], [-1,-1]]
        index = self.get_adjacent_node_index(node)

        min_index = index-1
        max_index = (index+1)%6
        if self.adjacent_nodes[min_index] != 0:
            if(self.adjacent_nodes[min_index].check_adjacent_node(node)):
                nr_triangles += 1
        if self.adjacent_nodes[max_index] != 0:
            if(self.adjacent_nodes[max_index].check_adjacent_node(node)):
                nr_triangles += 1
        return nr_triangles

    
        ## get the triangles in a specific edge
    def get_edge_triangles(self, node):
        triangles = []
        vectors = [[-1,0], [0,1], [1,1], [1,0], [0,-1], [-1,-1]]
        index = self.get_adjacent_node_index(node)

        min_index = index-1
        max_index = (index+1)%6
        if self.adjacent_nodes[min_index] != 0:
            if(self.adjacent_nodes[min_index].check_adjacent_node(node)):
                triangles += [[node, self, self.adjacent_nodes[min_index]],]
        if self.adjacent_nodes[max_index] != 0:
            if(self.adjacent_nodes[max_index].check_adjacent_node(node)):
                triangles += [[node, self, self.adjacent_nodes[max_index]],]
        return triangles


    ## count all the triangles around 1 node
    def count_triangles(self, node):
        nr_triangles = 0
        for index in [0, 2, 4]:
            if self.adjacent_nodes[index] != 0:
                nr_triangles += self.count_edge_triangles(self.adjacent_nodes[index])
        return nr_triangles


    def check_adjacent_node(self, edge):
        if(edge in self.adjacent_nodes):
            return True
        return False
    
    def get_adjacent_nodes(self):
        return [node for node in self.adjacent_nodes if node != 0]
    
    def to_string(self):
        print(f"node x:{self.x}, y:{self.y}")
    
    def get_simple_node(self):
        return [self.x, self.y]
    
    def get_simple_edges(self):
        edges = []
        for edge in self.adjacent_nodes:
            if(edge == 0):
                continue
            if(self.x > edge.x):
                edges = edges + [self.get_simple_node(), edge.get_simple_node()]
            elif self.x == edge.x:
                if(self.y > edge.y):
                    edges = edges + [[self.get_simple_node(), edge.get_simple_node()],]
        return edges
