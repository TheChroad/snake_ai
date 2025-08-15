import torch
import random
import gc
from collections import deque
from model import SnakeCNN, DQNTrainer

# Define hyperparameters
MAX_MEMORY = 30_000  # Maximum size of the replay memory deque
BATCH_SIZE = 256  # Number of experiences to sample from memory for long-term training
LR = 0.0005  # Learning rate for the optimizer


# Define the Agent class
class Agent:
    def __init__(self, board_size=5):
        # Initialize agent's state
        self.n_games = 0  # Number of games played
        self.epsilon = 0  # Epsilon for epsilon-greedy strategy
        self.gamma = 0.9  # Discount factor
        self.memory = deque(maxlen=MAX_MEMORY)  # Replay memory using deque
        self.board_size = board_size
        self.best_score = 0  # Best score achieved during training

        # Determine the device (GPU or CPU) for training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Using GPU: {gpu_name}")
            torch.cuda.empty_cache()  # Clear CUDA cache to free up memory
        else:
            print("💻 Using CPU (GPU not available)")

        # Initialize the deep Q-network model and the trainer
        self.model = SnakeCNN(board_size=board_size, input_channels=3, output_size=3)
        self.model.to(self.device)  # Move the model to the specified device
        self.trainer = DQNTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """
        Retrieves the current state of the game environment.
        """
        return game.get_state_matrix()

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple (state, action, reward, next_state, done) in the replay memory.
        Triggers a memory cleanup periodically.
        """
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) % 500 == 0:
            self._cleanup_memory()

    def _cleanup_memory(self):
        """
        Clears the CUDA cache and runs garbage collection to free up memory.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def train_long_memory(self):
        """
        Trains the model on a batch of experiences sampled from the replay memory.
        This is for long-term learning.
        """
        # Sample a mini-batch from the replay memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        if not mini_sample:
            return 0

        # Unpack the mini-batch into separate lists
        states = [s for s, a, r, ns, d in mini_sample]
        actions = [a for s, a, r, ns, d in mini_sample]
        rewards = [r for s, a, r, ns, d in mini_sample]
        next_states = [ns for s, a, r, ns, d in mini_sample]
        dones = [d for s, a, r, ns, d in mini_sample]

        try:
            # Perform a training step on the entire batch
            loss = self.trainer.train_batch(states, actions, rewards, next_states, dones)
            # Periodically clean up memory
            if self.n_games % 50 == 0:
                self._cleanup_memory()
            return loss
        except RuntimeError as e:
            print(f"Training error: {e}")
            self._cleanup_memory()
            return 0

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Trains the model on a single experience. This is for immediate learning after each step.
        """
        try:
            return self.trainer.train_step(state, action, reward, next_state, done)
        except RuntimeError as e:
            print(f"Short memory training error: {e}")
            self._cleanup_memory()
            return 0

    def get_action(self, state):
        """
        Selects an action based on the epsilon-greedy strategy.
        With a probability of epsilon, a random move is chosen. Otherwise, the model predicts the best move.
        """
        # Decrease epsilon as the number of games increases (exploration vs. exploitation)
        self.epsilon = max(10, 100 - self.n_games * 0.5)
        final_move = [0, 0, 0]

        # Exploration: choose a random move
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        # Exploitation: let the model predict the best move
        else:
            try:
                # Convert the state to a PyTorch tensor and prepare it for the model
                state_tensor = torch.tensor(state, dtype=torch.float)
                if len(state_tensor.shape) == 3:
                    state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)
                elif len(state_tensor.shape) == 2:
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)

                state_tensor = state_tensor.to(self.device)

                # Get the model's prediction without calculating gradients
                with torch.no_grad():
                    prediction = self.model(state_tensor)
                    # Get the index of the best action
                    move = torch.argmax(prediction).item()
                final_move[move] = 1
                del state_tensor  # Free up memory

            except RuntimeError as e:
                print(f"Action prediction error: {e}")
                self._cleanup_memory()
                # Fallback to a random move in case of an error
                move = random.randint(0, 2)
                final_move[move] = 1

        return final_move

    def save_model(self, file_path):
        """
        Saves the model's state, optimizer's state, and training statistics to a file.
        """
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'n_games': self.n_games,
                'best_score': self.best_score,
                'board_size': self.board_size
            }, file_path)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, file_path):
        """
        Loads a saved model from a file and restores its state and training statistics.
        """
        try:
            # Load the checkpoint from the file, specifying the device
            checkpoint = torch.load(file_path, map_location=self.device)
            # Get the board size from the checkpoint and handle potential mismatch
            board_size = checkpoint.get('board_size', 5)
            if board_size != self.board_size:
                print(f"Warning: Model board size ({board_size}) doesn't match current board size ({self.board_size})")
                self.model = SnakeCNN(board_size=board_size, input_channels=3, output_size=3)
                self.model.to(self.device)
                self.trainer = DQNTrainer(self.model, lr=LR, gamma=self.gamma)
                self.board_size = board_size
            # Load the model and optimizer state dictionaries
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Restore training statistics
            self.n_games = checkpoint.get('n_games', 0)
            self.best_score = checkpoint.get('best_score', 0)
            print(f"Model loaded from {file_path}")
            print(f"Loaded model: {self.n_games} games, best score: {self.best_score}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def set_eval_mode(self):
        """
        Sets the model to evaluation mode (e.g., disables dropout and batch normalization layers).
        Also sets epsilon to 0 to disable exploration.
        """
        self.model.eval()
        self.epsilon = 0

    def set_train_mode(self):
        """
        Sets the model to training mode.
        """
        self.model.train()

    def get_model_info(self):
        """
        Returns a dictionary with information about the current agent and model state.
        """
        # Calculate the number of parameters in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Get device information
        device_info = "GPU" if self.device.type == "cuda" else "CPU"
        if self.device.type == "cuda":
            device_info += f" ({torch.cuda.get_device_name(0)})"

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': device_info,
            'board_size': self.board_size,
            'games_played': self.n_games,
            'best_score': self.best_score,
            'memory_size': len(self.memory),
            'max_possible_score': self.board_size * self.board_size - 3
        }