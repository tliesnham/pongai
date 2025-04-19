import pygame as pg
import pandas as pd
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from pongmodel import PongModel
from cnn import SimpleCNN


class Paddle(pg.sprite.Sprite):
    def __init__(self, screen=(500, 500)):
        pg.sprite.Sprite.__init__(self)
        self.score = 0

        # Create paddle surface
        self.width = 5
        self.height = 40
        self.image = pg.Surface((self.width, self.height))
        self.image.fill((255, 255, 255))

        self.input_state = 0

        # Get rect and position
        self.rect = self.image.get_rect()

    def ball_collision(self, ball):
        """Check for collision with ball"""
        if self.rect.colliderect(ball.rect):
            ball.bounce()
            return True
        return False

    def out_of_bounds(self):
        """Check if the paddle is out of bounds"""
        court_line_width = 5
        if self.rect.top < court_line_width:
            self.rect.top = court_line_width
        elif self.rect.bottom > pg.display.get_surface().get_height() - court_line_width:
            self.rect.bottom = pg.display.get_surface().get_height() - court_line_width

class PlayerPaddle(Paddle):
    def __init__(self, screen=(500, 500)):
        super().__init__(screen=screen)
        self.rect.midright = (screen[0] - 20, screen[1] // 2)

    def update(self, dt, ball, **kwargs):
        self.input_state = 0
        if pg.key.get_pressed()[pg.K_w]:
            self.input_state = 1
            self.rect.y -= 500 * dt
        elif pg.key.get_pressed()[pg.K_s]:
            self.input_state = 2
            self.rect.y += 500 * dt
        self.out_of_bounds()

class AIPaddle(Paddle):
    def __init__(self, screen=(500, 500)):
        super().__init__(screen=screen)
        self.reaction_delay = 0  # milliseconds
        self.last_reaction_time = 0
        self.target_y = 0
        self.rect.midleft = (20, screen[1] // 2)

    def update(self, dt, ball, **kwargs):
        current_time = pg.time.get_ticks()
        if current_time - self.last_reaction_time > self.reaction_delay:
            self.target_y = ball.rect.centery
            self.last_reaction_time = current_time
        
        # Move towards the recorded target position
        if self.rect.centery < self.target_y:
            self.rect.y += 500 * dt  # Move down
        elif self.rect.centery > self.target_y:
            self.rect.y -= 500 * dt # Move up
        
        self.out_of_bounds()

class NNPaddle(Paddle):
    def __init__(self, screen=(500, 500)):
        super().__init__(screen=screen)
        self.screen = screen
        self.model = PongModel(5)
        self.model.load_state_dict(torch.load("models/pong_ai.pth"))
        self.model.eval()

        # Load the scalers
        import joblib
        self.scaler = joblib.load("scalers/scaler.gz")
        self.y_scaler = joblib.load("scalers/y_scaler.gz")

        self.target_y = 0
        self.rect.midleft = (20, self.screen[1] // 2)
    
    def update(self, dt, ball, player, **kwargs):
        data = pd.DataFrame({
            "ball_x": ball.rect.centerx,
            "ball_y": ball.rect.centery,
            "ball_velocity_x": ball.velocity[0],
            "ball_velocity_y": ball.velocity[1],
            "player_paddle_y": player.rect.centery
        }, index=[0])

        # Scale the input data
        scaled_data = self.scaler.transform(data)

        with torch.no_grad():
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
            scaled_prediction = self.model(input_tensor).item()

            # Create a DataFrame with the same column name used for y during training
            prediction_df = pd.DataFrame({"ai_paddle_y": scaled_prediction}, index=[0])
            # Inverse transform to get actual y position
            self.target_y = self.y_scaler.inverse_transform(prediction_df)[0]

        # Move towards the recorded target position
        if self.rect.centery < self.target_y:
            self.rect.y += 500 * dt  # Move down
        elif self.rect.centery > self.target_y:
            self.rect.y -= 500 * dt # Move up
        self.out_of_bounds()

class CNNPaddle(Paddle):
    def __init__(self, screen=(500, 500)):
        super().__init__(screen=screen)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN(num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load("models/pong_cnn.pth", map_location=self.device))
        self.model.eval()
        self.rect.midleft = (20, screen[1] // 2)

        # Match training transforms
        self.transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def get_action(self, game_state):
        img = Image.fromarray(game_state).convert('RGB')
        tensor_img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor_img)
            _, predicted = torch.max(output, 1)
        return predicted.item()

    def update(self, dt, frame, **kwargs):
        self.frame = frame
        self.frame = np.rot90(self.frame, k=3)
        self.frame = np.fliplr(self.frame)
        action = self.get_action(self.frame)
        
        # Move towards the recorded target position
        if action == 1:
            self.rect.y += 500 * dt  # Move down
        elif action == 2:
            self.rect.y -= 500 * dt # Move up
        self.out_of_bounds()

