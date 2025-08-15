import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Convolutional Neural Network (CNN) for the Snake AI
class SnakeCNN(nn.Module):
    def __init__(self, board_size=5, input_channels=3, output_size=3):
        """
        Initializes the CNN architecture.
        board_size: The size of the game board (e.g., 5 for a 5x5 board).
        input_channels: Number of input channels (e.g., 3 for a 3-channel state matrix).
        output_size: The number of possible actions (e.g., 3 for left, straight, right).
        """
        super(SnakeCNN, self).__init__()
        self.board_size = board_size

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)

        # Dropout layers to prevent overfitting
        self.dropout = nn.Dropout(0.15)

        # Fully connected layers
        # Calculate the size of the flattened tensor after the convolutional layers
        conv_output_size = 32 * board_size * board_size
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        # Initialize the weights of the network
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights and biases of the layers using appropriate methods.
        Kaiming normal initialization for convolutional and linear layers.
        Constant initialization for batch normalization layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        x: The input tensor representing the game state.
        """
        # Apply convolution, batch normalization, and ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Output layer (no activation for Q-values)
        return x


# Define the trainer for the Deep Q-Learning model
class DQNTrainer:
    def __init__(self, model, lr=0.002, gamma=0.95):
        """
        Initializes the trainer with a model, learning rate, and discount factor.
        model: The deep Q-learning model to be trained.
        lr: Learning rate for the optimizer.
        gamma: Discount factor for future rewards.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Get the device from the model's parameters
        self.device = next(model.parameters()).device
        # Use Adam optimizer with L2 regularization (weight_decay) and betas
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5, betas=(0.9, 0.999))
        # Use Mean Squared Error (MSE) as the loss function
        self.criterion = nn.MSELoss()
        # Learning rate scheduler to adjust the learning rate during training
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

    def train_step(self, state, action, reward, next_state, done):
        """
        Performs a single training step on one experience tuple.
        This is used for short-term memory training.
        """
        try:
            # Convert inputs to tensors and move them to the correct device
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)

            # Reshape the tensors to fit the model's input format (add a batch dimension and rearrange channels)
            if len(state.shape) == 3:
                state = state.permute(2, 0, 1).unsqueeze(0)
            elif len(state.shape) == 2:
                state = state.unsqueeze(0).unsqueeze(0)
            if len(next_state.shape) == 3:
                next_state = next_state.permute(2, 0, 1).unsqueeze(0)
            elif len(next_state.shape) == 2:
                next_state = next_state.unsqueeze(0).unsqueeze(0)

            state = state.to(self.device)
            next_state = next_state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)

            # Calculate the target Q-value
            with torch.no_grad():
                if done:
                    # If the game is over, the target is just the immediate reward
                    target = reward
                else:
                    # Otherwise, use the Bellman equation: target = reward + gamma * max(Q(next_state))
                    pred_next = self.model(next_state)
                    target = reward + self.gamma * torch.max(pred_next)

            # Get the current Q-values from the model
            pred = self.model(state)
            # Clone the prediction to create the target tensor for the loss calculation
            target_f = pred.clone()
            # Get the index of the taken action
            action_idx = torch.argmax(action).item()
            # Update the Q-value for the taken action with the calculated target
            target_f[0][action_idx] = target

            # Calculate the loss and perform backpropagation
            loss = self.criterion(target_f, pred)
            self.optimizer.zero_grad()  # Reset gradients
            loss.backward()  # Backpropagation
            # Clip gradients to prevent them from exploding
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()  # Update weights
            self.scheduler.step()  # Update learning rate

            return loss.item()
        except Exception as e:
            print(f"Training step error: {e}")
            return 0.0

    def train_batch(self, states, actions, rewards, next_states, dones):
        """
        Performs a training step on a batch of experiences.
        This is used for long-term memory training.
        """
        try:
            batch_size = len(states)
            if batch_size == 0:
                return 0.0

            # Convert lists of states and next_states into stacked tensors
            state_tensors = []
            next_state_tensors = []
            for i in range(batch_size):
                state = torch.tensor(states[i], dtype=torch.float)
                if len(state.shape) == 3:
                    state = state.permute(2, 0, 1)
                elif len(state.shape) == 2:
                    state = state.unsqueeze(0)
                state_tensors.append(state)

                next_state = torch.tensor(next_states[i], dtype=torch.float)
                if len(next_state.shape) == 3:
                    next_state = next_state.permute(2, 0, 1)
                elif len(next_state.shape) == 2:
                    next_state = next_state.unsqueeze(0)
                next_state_tensors.append(next_state)

            state_batch = torch.stack(state_tensors).to(self.device)
            next_state_batch = torch.stack(next_state_tensors).to(self.device)
            action_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
            reward_batch = torch.tensor(rewards, dtype=torch.float).to(self.device)
            done_batch = torch.tensor(dones, dtype=torch.bool).to(self.device)

            # Get Q-values for the current states
            current_q_values = self.model(state_batch)

            # Calculate the target Q-values using the Bellman equation
            with torch.no_grad():
                next_q_values = self.model(next_state_batch)
                # Get the maximum Q-value for each next state
                max_next_q_values = torch.max(next_q_values, dim=1)[0]
                # Calculate the target Q-value, setting it to the reward if the game is done
                target_q_values = reward_batch + (self.gamma * max_next_q_values * ~done_batch)

            # Select the Q-values corresponding to the actions taken
            action_indices = torch.argmax(action_batch, dim=1)
            predicted_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

            # Calculate the loss and perform backpropagation
            loss = self.criterion(predicted_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.scheduler.step()

            return loss.item()
        except Exception as e:
            print(f"Batch training error: {e}")
            return 0.0

    def get_lr(self):
        """
        Returns the current learning rate of the optimizer.
        """
        return self.optimizer.param_groups[0]['lr']