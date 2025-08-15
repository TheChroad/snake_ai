import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple


# Enumeration for the direction of the snake
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Named tuple for a point on the board
Point = namedtuple('Point', 'x, y')

# Color definitions
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Game constants
BLOCK_SIZE = 80
SPEED = 5


class SnakeGameAI:
    def __init__(self, w=5, h=5, render=True):
        """
        Initializes the Snake Game environment.
        w, h: Dimensions of the game board.
        render: Boolean to enable/disable GUI rendering.
        """
        self.w = w
        self.h = h
        self.render = render
        self.prev_distance_to_food = 0
        self.steps_without_food = 0
        self.display = None
        self.clock = None
        self.font = None

        # Initialize Pygame display if rendering is enabled
        if self.render:
            self._init_display()

        # Reset the game state to its initial configuration
        self.reset()

    def _init_display(self):
        """Initializes the Pygame display and related components."""
        try:
            # Check if Pygame has been initialized
            if not pygame.get_init():
                pygame.init()

            self.font = pygame.font.Font(None, 25)
            # Set up the display with a buffer for the score
            self.display = pygame.display.set_mode((self.w * BLOCK_SIZE, self.h * BLOCK_SIZE + 100))
            pygame.display.set_caption('Snake AI')
            self.clock = pygame.time.Clock()

        except pygame.error as e:
            print(f"Pygame display initialization failed: {e}")
            self.render = False  # Disable rendering if initialization fails

    def reset(self):
        """
        Resets the game to its initial state for a new game.
        """
        self.direction = Direction.RIGHT
        # Place the snake head in the center of the board
        self.head = Point(self.w // 2, self.h // 2)
        # The snake starts with two segments
        self.snake = [self.head, Point(self.head.x - 1, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()  # Place the first food item
        self.frame_iteration = 0  # Counter for game steps
        self.steps_without_food = 0  # Counter for steps taken without eating
        self.prev_distance_to_food = self._get_distance_to_food()  # Initial distance for reward calculation

    def _place_food(self):
        """
        Places the food item at a random empty location on the board.
        """
        x = random.randint(0, self.w - 1)
        y = random.randint(0, self.h - 1)
        self.food = Point(x, y)
        # Ensure food is not placed on the snake's body
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Performs one step in the game based on the agent's action.
        action: A list representing the chosen action [straight, right, left].
        Returns: reward, game_over flag, and current score.
        """
        self.frame_iteration += 1
        self.steps_without_food += 1

        # Handle Pygame events if rendering is enabled
        if self.render:
            if self.display is None or pygame.display.get_surface() is None:
                self._init_display()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.display = None
                    return -10, "quit", self.score  # Special return for quitting the training process

        # Move the snake based on the action
        self._move(action)
        # Insert the new head at the front of the snake list
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        # Check for collision
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if the snake has eaten the food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.steps_without_food = 0
            self._place_food()  # Place new food
        else:
            # If no food is eaten, remove the tail of the snake
            self.snake.pop()
            new_distance = self._get_distance_to_food()
            # Reward for getting closer to the food
            if new_distance < self.prev_distance_to_food:
                reward = 1
            else:
                reward = -1  # Penalty for getting further away

            self.prev_distance_to_food = new_distance

            # Penalty for not eating for too long
            if self.steps_without_food > 50:
                reward = -5
                game_over = True

        # Update the UI if rendering is enabled
        if self.render and self.display is not None:
            self._update_ui()
            if self.clock:
                self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Checks for a collision with the boundaries or the snake's own body.
        pt: The point to check for collision. Defaults to the snake's head.
        """
        if pt is None:
            pt = self.head
        # Check if the point is outside the board boundaries
        if pt.x > self.w - 1 or pt.x < 0 or pt.y > self.h - 1 or pt.y < 0:
            return True
        # Check if the point is in the snake's body
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """
        Renders the game state on the Pygame display.
        """
        if self.display is None:
            return

        try:
            self.display.fill(BLACK)

            # Draw the snake
            for i, pt in enumerate(self.snake):
                color = BLUE1 if i == 0 else BLUE2  # Head is a different color
                pygame.draw.rect(self.display, color,
                                 pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, WHITE,
                                 pygame.Rect(pt.x * BLOCK_SIZE + 2, pt.y * BLOCK_SIZE + 2, BLOCK_SIZE - 4,
                                             BLOCK_SIZE - 4),
                                 2)

            # Draw the food
            pygame.draw.rect(self.display, RED,
                             pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, YELLOW,
                             pygame.Rect(self.food.x * BLOCK_SIZE + 4, self.food.y * BLOCK_SIZE + 4, BLOCK_SIZE - 8,
                                         BLOCK_SIZE - 8))

            # Draw the status bar at the bottom
            pygame.draw.rect(self.display, (30, 30, 30), pygame.Rect(0, self.h * BLOCK_SIZE, self.w * BLOCK_SIZE, 100))
            max_score = self.get_max_possible_score()
            progress = (self.score / max_score) * 100 if max_score > 0 else 0

            # Render score and step text
            if self.font:
                score_text = self.font.render(f"Score: {self.score}/{max_score} ({progress:.1f}%)", True, GREEN)
                steps_text = self.font.render(f"Steps: {self.frame_iteration}", True, GREEN)

                score_x = (self.w * BLOCK_SIZE - score_text.get_width()) // 2
                steps_x = (self.w * BLOCK_SIZE - steps_text.get_width()) // 2

                self.display.blit(score_text, (score_x, self.h * BLOCK_SIZE + 5))
                self.display.blit(steps_text, (steps_x, self.h * BLOCK_SIZE + 30))

            pygame.display.flip()  # Update the full display Surface to the screen

        except pygame.error as e:
            print(f"UI update error: {e}")
            self.display = None

    def _move(self, action):
        """
        Updates the snake's direction and head position based on the action.
        action: [straight, right, left]
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  # Go straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # Turn right (clockwise)
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # Turn left (counter-clockwise)
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir
        x = self.head.x
        y = self.head.y

        # Update head coordinates based on the new direction
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)

    def get_state_matrix(self):
        """
        Creates a 3D NumPy array representing the game state.
        Channel 0: Snake body.
        Channel 1: Snake head.
        Channel 2: Food.
        """
        # Initialize a zero matrix with dimensions (height, width, channels)
        state = np.zeros((self.h, self.w, 3), dtype=np.float32)

        # Mark snake body
        for pt in self.snake[1:]:
            if 0 <= pt.x < self.w and 0 <= pt.y < self.h:
                state[pt.y, pt.x, 0] = 1.0

        # Mark snake head
        if 0 <= self.head.x < self.w and 0 <= self.head.y < self.h:
            state[self.head.y, self.head.x, 1] = 1.0

        # Mark food
        if 0 <= self.food.x < self.w and 0 <= self.food.y < self.h:
            state[self.food.y, self.food.x, 2] = 1.0

        return state

    def get_max_possible_score(self):
        """
        Calculates the maximum possible score for the current board size.
        """
        return self.w * self.h - 2

    def _get_distance_to_food(self):
        """
        Calculates the Manhattan distance from the snake's head to the food.
        """
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)