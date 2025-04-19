import pygame as pg
import csv

from paddle import AIPaddle, PlayerPaddle
from ponglogger import PongLogger
from ball import Ball
from utils import draw_score, draw_court

LOG_DATA = True

def main():
    logger = PongLogger(screenshots_dir="screenshots", data_file="data/pong_data.csv")

    pg.init()
    screen = pg.display.set_mode((600, 600), vsync=1)
    pg.display.set_caption("Pong")
    pg.mouse.set_visible(False)

    background = pg.Surface(screen.get_size()).convert()
    background.fill((0, 0, 0))

    screen.blit(background, (0, 0))
    pg.display.flip()

    # Create paddles and ball
    player = PlayerPaddle(screen=screen.get_size())
    ai = AIPaddle(screen=screen.get_size())
    ball = Ball(radius=4, screen=screen.get_size())
    allsprites = pg.sprite.RenderPlain((player, ai, ball))
    clock = pg.time.Clock()

    running = True
    rally = 0
    while running:
        dt = clock.tick(60) / 1000.0

        if LOG_DATA:
            if frame % 6 == 0: # Log every 6th frame
                logger.log_frame(player, ai, ball, screen)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                running = False
            
            if event.type == Ball.GOAL_EVENT:
                if event.scorer == "player":
                    player.score += 1
                    rally = 0
                elif event.scorer == "ai":
                    ai.score += 1
                    rally = 0
        
        if player.ball_collision(ball):
            rally += 1
        elif ai.ball_collision(ball):
            rally += 1
        
        # Only apply rally modifer after 5 rallies
        # This prevents the ball from getting too fast at the start
        if rally >= 3:
            ball.rally_modifier = min(1.0 + (rally * 0.03), 2.5)
        
        frame = pg.surfarray.array3d(pg.display.get_surface())
        allsprites.update(dt=dt, ball=ball, player=player, frame=frame)

        screen.blit(background, (0, 0))
        draw_court(screen)
        draw_score(screen, player.score, ai.score)
        allsprites.draw(screen)
        pg.display.flip()

    if LOG_DATA:
        logger.save_data()
    pg.quit()

if __name__ == "__main__":
    main()