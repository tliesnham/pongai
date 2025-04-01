import pygame as pg
import csv

from paddle import AIPaddle, PlayerPaddle
from ball import Ball


def save_to_csv(data_list, filename="data/pong_data.csv"):
    with open(filename, mode='w', newline='') as csvfile:
        fieldnames = [
            "ai_paddle_y",
            "player_paddle_y",
            "ball_x",
            "ball_y",
            "ball_velocity_x",
            "ball_velocity_y",
            "player_score",
            "ai_score",
            "rally_modifier"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_list:
            writer.writerow(data)

def log_data(data_list, player, ai, ball):
    data_list.append({
        "ai_paddle_y": ai.rect.centery,
        "player_paddle_y": player.rect.centery,
        "ball_x": ball.rect.x,
        "ball_y": ball.rect.y,
        "ball_velocity_x": ball.velocity[0],
        "ball_velocity_y": ball.velocity[1],
        "player_score": player.score,
        "ai_score": ai.score,
        "rally_modifier": ball.rally_modifier,
    })

def draw_score(screen, player_score, ai_score):
    font = pg.font.SysFont("Arial", 36)
    player_text = font.render(f"Player: {player_score}", True, (255, 255, 255))
    ai_text = font.render(f"AI: {ai_score}", True, (255, 255, 255))

    screen.blit(player_text, (screen.get_width() - ai_text.get_width() - 80, 10))
    screen.blit(ai_text, (10, 10))

def draw_court(screen):
    screen_width, screen_height = screen.get_size()
    dash_length = 10
    for y in range(0, screen_height, 20):
        pg.draw.line(
            screen,
            (150, 150, 150),
            (screen_width // 2, y),
            (screen_width // 2, y + dash_length),
            3
        )

    # Draw top and bottom lines
    pg.draw.line(screen, (150, 150, 150), (0, 0), (screen_width, 0), 5)
    pg.draw.line(screen, (150, 150, 150), (0, screen_height), (screen_width, screen_height), 5)

def main():
    data_list = []
    pg.init()
    screen = pg.display.set_mode((800, 600), pg.SCALED, vsync=1)
    pg.display.set_caption("Pong")
    pg.mouse.set_visible(False)

    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill((0, 0, 0))

    screen.blit(background, (0, 0))
    pg.display.flip()

    # Create paddles and ball
    player = PlayerPaddle(screen=screen.get_size())
    ai = AIPaddle(screen=screen.get_size())
    ball = Ball(radius=10, screen=screen.get_size())
    allsprites = pg.sprite.RenderPlain((player, ai, ball))
    clock = pg.time.Clock()

    running = True
    rally = 0
    while running:
        dt = clock.tick(120) / 1000.0
        log_data(data_list, player, ai, ball)

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
            ball.rally_modifier = min(1.0 + (rally * 0.03), 2.0)
        
        allsprites.update(dt=dt, ball=ball)

        screen.blit(background, (0, 0))
        draw_court(screen)
        draw_score(screen, player.score, ai.score)
        allsprites.draw(screen)
        pg.display.flip()

    save_to_csv(data_list)
    pg.quit()

if __name__ == "__main__":
    main()