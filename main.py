import os
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog

import matplotlib
import matplotlib.pyplot as plt
import pygame

# Import custom classes for the Snake AI environment and the agent
from snake_env import SnakeGameAI
from agent import Agent


class SnakeAIGUI:
    def __init__(self):
        # Initialize Pygame for game visualization
        pygame.init()
        # Set up a font for displaying text in Pygame
        self.font = pygame.font.Font(None, 25)

        # Initialize the main Tkinter window
        self.root = tk.Tk()
        self.root.title("Snake AI - Training & Playing")
        self.root.geometry("420x320")
        self.root.configure(bg="#2c3e50")

        # Define directories for saving models and plots, and create them if they don't exist
        self.models_dir = "models"
        self.plots_dir = "plots"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Flags to track the state of the application
        self.is_training = False
        self.training_thread = None
        self.play_thread = None

        # Set up the graphical user interface
        self._setup_gui()

    def _add_hover_effect(self, widget, normal_color, hover_color):
        """
        Adds a hover effect to a Tkinter widget, changing its background color
        when the mouse enters or leaves.
        """

        def on_enter(e):
            widget["bg"] = hover_color

        def on_leave(e):
            widget["bg"] = normal_color

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _setup_gui(self):
        """
        Sets up the main layout and widgets of the GUI.
        """
        # Title label
        title = tk.Label(
            self.root, text="🐍 Snake AI",
            font=("Arial", 24, "bold"),
            bg="#2c3e50", fg="#ecf0f1"
        )
        title.pack(pady=(15, 5))

        # Description label
        desc = tk.Label(
            self.root, text="Choose an option:",
            font=("Arial", 12),
            bg="#2c3e50", fg="#bdc3c7"
        )
        desc.pack(pady=(0, 10))

        # Frame for the buttons
        frame = tk.Frame(self.root, bg="#2c3e50")
        frame.pack(pady=10)

        # Style dictionary for the buttons
        btn_style = {
            "font": ("Arial", 12, "bold"),
            "width": 14,
            "height": 2,
            "bd": 0,
            "relief": "flat",
            "activeforeground": "white"  # Text color on click
        }

        # Train button
        train_btn = tk.Button(
            frame, text="Train", bg="#3498db", fg="white",
            command=self._on_train_clicked, **btn_style
        )
        train_btn.grid(row=0, column=0, padx=8, pady=6)

        # Play button
        play_btn = tk.Button(
            frame, text="Play", bg="#2ecc71", fg="white",
            command=self._on_play_clicked, **btn_style
        )
        play_btn.grid(row=0, column=1, padx=8, pady=6)

        # Add hover effects to the buttons
        self._add_hover_effect(train_btn, "#3498db", "#5dade2")
        self._add_hover_effect(play_btn, "#2ecc71", "#58d68d")

        # Label to display available models
        self.models_label = tk.Label(
            self.root, font=("Arial", 9),
            bg="#2c3e50", fg="#7f8c8d", wraplength=380
        )
        self.models_label.pack(pady=8)
        self._update_models_display()

        # Status label to show the current application status
        self.status_label = tk.Label(
            self.root, text="Ready",
            font=("Arial", 10),
            bg="#2c3e50", fg="#95a5a6"
        )
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    def _get_available_models(self):
        """
        Retrieves a list of available model file names from the models directory.
        """
        return [f[:-4] for f in os.listdir(self.models_dir) if f.endswith(".pth")]

    def _update_models_display(self):
        """
        Updates the GUI label with the list of currently available models.
        """
        models = self._get_available_models()
        text = "Available models: " + ", ".join(models) if models else "No models available"
        self.models_label.config(text=text)

    def _on_train_clicked(self):
        """
        Handler for the 'Train' button click. Prompts the user for training parameters.
        """
        if self.is_training:
            messagebox.showwarning("Training in progress", "A training is already running.")
            return

        # Ask the user for the number of games to train
        num_games = simpledialog.askinteger("Training Setup", "How many games to train?", minvalue=1,
                                            maxvalue=500000, initialvalue=1000, parent=self.root)
        if num_games is None:
            return

        # Ask if the user wants to train in headless mode (without visualization)
        headless = messagebox.askyesno("Fast Training", "Train without visualization? (much faster)",
                                       parent=self.root)
        self._start_training(num_games, headless)

    def _start_training(self, num_games, headless):
        """
        Starts the training process in a separate thread.
        """
        self.is_training = True
        self.status_label.config(text=f"Training in progress... (0/{num_games})")
        # Create a new thread to run the training process so the GUI doesn't freeze
        self.training_thread = threading.Thread(target=self._train_agent, args=(num_games, headless), daemon=True)
        self.training_thread.start()

    def _train_agent(self, num_games, headless):
        """
        The main training loop for the AI agent. This function runs in a separate thread.
        """
        try:
            agent = Agent()
            game = SnakeGameAI(render=not headless)
            scores = []
            mean_scores = []
            total_score = 0
            record = 0

            # Main training loop
            while agent.n_games < num_games and self.is_training:
                # Get old state
                state_old = agent.get_state(game)
                # Get move based on the current state
                final_move = agent.get_action(state_old)
                # Perform the move and get new state
                reward, done, score = game.play_step(final_move)
                state_new = agent.get_state(game)

                # Handle quit event
                if done == "quit":
                    self.root.after(0, lambda: self.status_label.config(text="Training cancelled by user"))
                    self.root.after(0, lambda: messagebox.showinfo("Training Cancelled",
                                                                   f"Training was cancelled.\nGames completed: {agent.n_games}\nBest score: {record}",
                                                                   parent=self.root))
                    return

                # Train the short memory of the agent (Q-learning)
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                # Store the experience in the replay memory
                agent.remember(state_old, final_move, reward, state_new, done)

                # If the game is over
                if done and done != "quit":
                    game.reset()
                    agent.n_games += 1

                    # Train the long memory every 3 games
                    if agent.n_games % 3 == 0:
                        loss = agent.train_long_memory()

                    # Update the high score
                    if score > record:
                        record = score
                        agent.best_score = record

                    scores.append(score)
                    total_score += score
                    mean_scores.append(total_score / agent.n_games)

                    # Update the status label in the GUI every 25 games or at the end
                    if agent.n_games % 25 == 0 or agent.n_games == num_games:
                        avg_score = total_score / agent.n_games
                        self.root.after(0, lambda g=agent.n_games, s=score, r=record, avg=avg_score:
                        self.status_label.config(
                            text=f"Training... ({g}/{num_games}) - Score: {s}, Record: {r}, Avg: {avg:.1f}"
                        ))

            # Save the trained model
            model_name = f"model_{num_games}"
            model_path = os.path.join(self.models_dir, model_name + ".pth")
            agent.save_model(model_path)

            # Create and save a plot of the training progress
            try:
                matplotlib.use('Agg')  # Use non-interactive backend for saving
                plt.figure(figsize=(10, 6))

                plot_scores = scores
                plot_means = mean_scores

                plt.plot(plot_scores, label='Scores', alpha=0.6)
                plt.plot(plot_means, label='Mean Score', linewidth=2)
                plt.title(f'Training Progress - {num_games} Games')
                plt.xlabel('Game')
                plt.ylabel('Score')
                plt.legend()
                plt.grid(alpha=0.3)

                plot_path = os.path.join(self.plots_dir, f"plot_{model_name}.png")
                plt.savefig(plot_path, dpi=200, bbox_inches='tight')
                plt.close()

            except Exception as plot_error:
                print(f"Plot creation failed: {plot_error}")
                plot_path = "Plot creation failed"

            # Call the completion handler on the main GUI thread
            self.root.after(0, lambda: self._on_training_completed(model_name, record, model_path, plot_path))

        except Exception as e:
            print(f"Training error: {e}")
            self.root.after(0, lambda e=e: messagebox.showerror("Error", f"Training Error: {e}"))
        finally:
            # Clean up and reset flags
            self.is_training = False
            try:
                pygame.quit()
            except:
                pass

    def _on_training_completed(self, model_name, record, model_path, plot_path):
        """
        Handler for when the training process finishes. Displays a message and updates the GUI.
        """
        self.status_label.config(text="Training complete.")
        messagebox.showinfo("Finished", f"Model saved: {model_path}\nPlot: {plot_path}\nRecord: {record}",
                            parent=self.root)
        self._update_models_display()

    def _on_play_clicked(self):
        """
        Handler for the 'Play' button click. Prompts the user to select a model.
        """
        models = self._get_available_models()
        if not models:
            messagebox.showwarning("No models", "No models available.", parent=self.root)
            return

        # Ask the user for the model name to play with
        model_name = simpledialog.askstring("Model", f"Enter model name:\nAvailable: {', '.join(models)}",
                                            parent=self.root)
        if not model_name:
            return

        # Check if the model file exists
        model_path = os.path.join(self.models_dir, model_name + ".pth")
        if not os.path.exists(model_path):
            messagebox.showerror("Not found", f"Model '{model_name}' not found.", parent=self.root)
            return

        # Start the game loop in a new thread
        self.play_thread = threading.Thread(target=self._play_with_model, args=(model_path,), daemon=True)
        self.play_thread.start()

    def _play_with_model(self, model_path):
        """
        The game loop for playing with a trained AI model. This runs in a separate thread.
        """
        try:
            # Re-initialize Pygame for the game window
            pygame.quit()
            pygame.init()

            # Load the agent and the trained model
            agent = Agent()
            agent.load_model(model_path)
            agent.set_eval_mode()

            game = SnakeGameAI(render=True)
            self.root.after(0, lambda: self.status_label.config(text="Playing (close Pygame window to stop)"))

            running = True
            while running:
                # Event handling for closing the window
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break

                # Get the current state of the game
                state = agent.get_state(game)
                # Get the action from the AI agent
                final_move = agent.get_action(state)
                # Play the step and get feedback
                reward, done, score = game.play_step(final_move)

                # Control the game speed
                if game.clock:
                    game.clock.tick(20)

                # If the game is over
                if done:
                    self.root.after(0, lambda s=score: self.status_label.config(text=f"Game Over! Score: {s}"))

                    # Wait for a moment before showing the next dialog
                    pygame.time.wait(800)
                    pygame.quit()  # Close the Pygame window

                    # Ask if the user wants to play again
                    again = messagebox.askyesno("Play again?", f"Score: {score}\nPlay again?", parent=self.root)
                    if again:
                        pygame.init()  # Re-initialize for a new game
                        game = SnakeGameAI(render=True)
                        continue  # Start a new game loop
                    else:
                        running = False
                        break  # Exit the game loop

        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("Error", f"Play Error: {e}"))
        finally:
            # Clean up and reset status
            try:
                pygame.quit()
            except:
                pass
            self.root.after(0, lambda: self.status_label.config(text="Ready"))

    def run(self):
        """
        Starts the Tkinter main event loop.
        """
        self.root.mainloop()


if __name__ == "__main__":
    app = SnakeAIGUI()
    app.run()