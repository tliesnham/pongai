import pygame as pg
import random


class Ball(pg.sprite.Sprite):
    GOAL_EVENT = pg.USEREVENT + 1

    def __init__(self, radius=10, screen=(800, 600)):
        pg.sprite.Sprite.__init__(self)
        self.rally_modifier = 1.0
        # Screen dimensions
        self.screen_width = screen[0]
        self.screen_height = screen[1]

        # Create transparent surface for the circle
        self.radius = radius
        self.image = pg.Surface((2 * self.radius, 2 * self.radius), pg.SRCALPHA)

        # Draw red circle to the surface
        pg.draw.circle(
            self.image,
            (255, 0, 0),
            (self.radius, self.radius),
            self.radius
        )

        # Get rect and position in centre
        self.rect = self.image.get_rect(center=(self.screen_width // 2, self.screen_height // 2))

        # Add movement attributes
        self.velocity = [300, 0]  # Horizontal movement only when spawned, player starts
        
    def update(self, dt, **kwargs):
        """Basic movement logic"""
        self.rect.x += self.velocity[0] * dt
        self.rect.y += self.velocity[1] * dt
        
        # Bounce off top and bottom
        if self.rect.y <= 0:
            self.velocity[1] = max(1, abs(self.velocity[1]))  # Ensure it moves down
        elif self.rect.y >= self.screen_height - self.radius * 2:
            self.velocity[1] = min(-1, -abs(self.velocity[1]))

        # Check for goal
        if self.rect.left <= 30:
            pg.event.post(pg.event.Event(Ball.GOAL_EVENT, scorer="player"))
            self.reset()
        elif self.rect.right >= self.screen_width - 30:
            pg.event.post(pg.event.Event(Ball.GOAL_EVENT, scorer="ai"))
            self.reset()
    
    def reset(self):
        """Recentre after point"""
        self.rect.center = (self.screen_width // 2, self.screen_height // 2)
        self.velocity = [random.choice([-300, 300]), 0]
    
    def bounce(self):
        """Reverse and randomise the y velocity"""
        if self.velocity[0] > 0:
            self.velocity[0] = -300
        else:
            self.velocity[0] = 300
        self.velocity[0] *= self.rally_modifier
        self.velocity[1] = random.randint(-400, 400) * self.rally_modifier
       
        if self.velocity[1] == 0:
            self.velocity[1] = 1 * self.rally_modifier