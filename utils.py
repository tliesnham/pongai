import pygame as pg
import csv

def draw_score(screen, player_score, ai_score):
    font = pg.font.SysFont("Arial", 18)
    player_text = font.render(f"Player: {player_score}", True, (255, 255, 255))
    ai_text = font.render(f"AI: {ai_score}", True, (255, 255, 255))

    screen.blit(player_text, (screen.get_width() - ai_text.get_width() - 50, 10))
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