# 🐍 Snake AI - Deep Q-Network Training & Playing

A sophisticated Snake AI implementation using Deep Q-Network (DQN) with Convolutional Neural Networks. Train your own AI agent to play Snake and watch it improve over time!

## ✨ Features

- **Deep Q-Network (DQN)** with Convolutional Neural Network
- **GPU Acceleration** (CUDA support for NVIDIA GPUs)
- **Interactive GUI** for training and playing
- **Real-time Visualization** during training and gameplay
- **Model Persistence** - Save and load trained models
- **Training Progress Plots** with matplotlib
- **Multi-threading** for responsive GUI
- **Cross-platform** compatibility (Windows, Linux, macOS)

## 🎮 Demo

The AI learns to play Snake by:
1. Observing the game state as a 3D matrix (snake body, head, food)
2. Predicting Q-values for each action (straight, right, left)
3. Learning from rewards and penalties
4. Improving performance over thousands of games

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (optional, but recommended for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/snake-ai.git
   cd snake-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup GPU support (optional but recommended)**
   
   Check your CUDA version:
   ```bash
   nvidia-smi
   ```
   
   Install GPU-enabled PyTorch:
   ```bash
   # For CUDA 12.x (RTX 30/40 series with recent drivers)
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.x or maximum compatibility
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## 🎯 Usage

### Training a Model
1. Click **"Train"** in the GUI
2. Set number of games (recommended: 1000+ for good results)
3. Choose visualization mode:
   - **With visualization**: Slower but you can watch the AI learn
   - **Headless mode**: Much faster training without graphics
4. Wait for training to complete
5. Model and training plot will be saved automatically

### Playing with a Trained Model
1. Click **"Play"** in the GUI
2. Enter the name of a saved model
3. Watch your AI play Snake!
4. Close the game window to stop

### Model Files
- Trained models are saved in `models/` directory
- Training plots are saved in `plots/` directory
- Models include metadata (games played, best score, etc.)

## 🧠 Architecture

### Neural Network
- **Input**: 5×5×3 game state matrix (configurable board size)
  - Channel 0: Snake body positions
  - Channel 1: Snake head position  
  - Channel 2: Food position
- **Architecture**: CNN with 3 convolutional layers + 3 fully connected layers
- **Output**: Q-values for 3 actions (straight, turn right, turn left)

### Training Algorithm
- **Deep Q-Network (DQN)** with experience replay
- **Epsilon-greedy exploration** with decay
- **Reward system**:
  - +10 for eating food
  - +1 for moving closer to food
  - -1 for moving away from food
  - -10 for collision (game over)
  - -5 for taking too long without food

### Key Components
- `SnakeGameAI`: Game environment with matrix state representation
- `Agent`: DQN agent with experience replay and training logic
- `SnakeCNN`: Convolutional neural network model
- `DQNTrainer`: Training loop with loss computation and optimization

## 📁 Project Structure

```
snake-ai/
├── main.py              # GUI application entry point
├── snake_env.py         # Game environment
├── agent.py             # DQN agent implementation
├── model.py             # Neural network architecture
├── requirements.txt     # Dependencies
├── models/              # Saved models directory
├── plots/               # Training plots directory
└── README.md           # This file
```

## ⚙️ Configuration

Key hyperparameters in `agent.py`:
- `MAX_MEMORY`: 30,000 (experience replay buffer size)
- `BATCH_SIZE`: 256 (training batch size)
- `LR`: 0.0005 (learning rate)
- `GAMMA`: 0.9 (discount factor)

Game settings in `snake_env.py`:
- `BLOCK_SIZE`: 80 (pixel size of game blocks)
- `SPEED`: 5 (game speed for visualization)

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Windows 10, Linux (Ubuntu 18.04+), macOS 10.15+
- **RAM**: 4GB
- **Python**: 3.8+
- **Storage**: 500MB free space

### Recommended for GPU Training
- **GPU**: NVIDIA GTX 1060 or better
- **RAM**: 8GB+
- **CUDA**: 11.8 or 12.x
- **VRAM**: 4GB+

## 🐛 Troubleshooting

### GPU Not Detected
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation
3. Reinstall PyTorch with correct CUDA version
4. Restart terminal/IDE after installation

### Performance Issues
- Reduce batch size if running out of memory
- Use headless training mode for faster training
- Close other GPU-intensive applications

### Import Errors
1. Ensure Python 3.8+ is installed
2. Update pip: `pip install --upgrade pip`
3. Try using a virtual environment
4. Check all dependencies are installed correctly

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

# 📚 License

Content is provided under the [MIT License](https://opensource.org/licenses/MIT) unless otherwise noted.

---

**Happy training! 🚀** If you create an interesting AI model, share your results!
