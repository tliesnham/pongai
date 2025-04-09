import os
import csv
import pygame as pg


class PongLogger:
    def __init__(self, screenshots_dir="screenshots", data_file="data/pong_data.csv"):
        self.screenshots_dir = screenshots_dir
        self.data_file = data_file
        self.data_list = []
        self.frame = 0

        self._ensure_directories()
    
    def _ensure_directories(self):
        # Create directories if they do not exist
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)

    def log_frame(self, player, ai, ball, screen):
        self.frame += 1

        try:
            screenshot_path = f"{self.screenshots_dir}/frame-{self.frame}.jpeg"
            pg.image.save(screen, screenshot_path)

            self.data_list.append({
                "ai_paddle_y": ai.rect.centery,
                "player_paddle_y": player.rect.centery,
                "ball_x": ball.rect.x,
                "ball_y": ball.rect.y,
                "ball_velocity_x": ball.velocity[0],
                "ball_velocity_y": ball.velocity[1],
                "player_score": player.score,
                "ai_score": ai.score,
                "rally_modifier": ball.rally_modifier,
                "screenshot": screenshot_path,
                "frame": self.frame,
                "player_state": player.input_state
            })
        except:
            print(f"Error saving screenshot for frame {self.frame}")

    def save_data(self):
        if not self.data_list:
            print("No data to save.")
            return
        
        try:
            with open(self.data_file, mode='w', newline='') as csvfile:
                fieldnames = [
                    "ai_paddle_y",
                    "player_paddle_y",
                    "ball_x",
                    "ball_y",
                    "ball_velocity_x",
                    "ball_velocity_y",
                    "player_score",
                    "ai_score",
                    "rally_modifier",
                    "screenshot",
                    "frame",
                    "player_state"
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for data in self.data_list:
                    writer.writerow(data)
            print(f"Successfully saved data to {self.data_file}")
        except Exception as e:
            print(f"Error saving data: {e}")