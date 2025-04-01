import pygame as pg


class Paddle(pg.sprite.Sprite):
    def __init__(self, player=False, screen=(800, 600)):
        pg.sprite.Sprite.__init__(self)
        self.score = 0
        self.reaction_delay = 200  # milliseconds
        self.last_reaction_time = 0
        self.target_y = 0
        self.player = player

        # Create paddle surface
        self.width = 20
        self.height = 100
        self.image = pg.Surface((self.width, self.height))
        self.image.fill((255, 255, 255))

        # Get rect and position
        self.rect = self.image.get_rect()

        # If player spawn on right hand side
        if player:
            self.rect.midright = (screen[0] - 20, screen[1] // 2)
        else:
            self.rect.midleft = (20, screen[1] // 2)
    
    def update_ai(self, ball, dt):
        current_time = pg.time.get_ticks()
        if current_time - self.last_reaction_time > self.reaction_delay:
            self.target_y = ball.rect.centery
            self.last_reaction_time = current_time
        
        # Move towards the recorded target position
        if self.rect.centery < self.target_y:
            self.rect.y += 500 * dt  # Move down
        elif self.rect.centery > self.target_y:
            self.rect.y -= 500 * dt # Move up
    
    def update(self, dt, ball, **kwargs):
        if self.player:
            if pg.key.get_pressed()[pg.K_w]:
                self.rect.y -= 500 * dt
            elif pg.key.get_pressed()[pg.K_s]:
                self.rect.y += 500 * dt
        else:
            self.update_ai(ball, dt)
        self.out_of_bounds()

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