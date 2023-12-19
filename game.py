import random
import math
import time
import pygame
from pygame.locals import *
import numpy as np

# Initialize the game engine
pygame.init()

# Define the colors we will use in RGB format
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 205)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
DARK_BLUE = (25, 25, 112)
bright_red = (255, 0, 0)
bright_green = (0, 255, 0)

background = pygame.image.load('space.jpg')

# Set the height and width of the screen
screen_size = [580, 800]
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Game")
clock = pygame.time.Clock()

# audio variable
crash_sound = pygame.mixer.Sound("crash2.wav")


def create_ship(x, y, size):
    ship = pygame.image.load('ship2.png')
    ship = pygame.transform.scale(ship, (size, size))
    screen.blit(ship, (x, y))


def create_asteroid(x, y, size):
    asteroid = pygame.image.load('asteroid.png')
    asteroid = pygame.transform.scale(asteroid, (size, size))
    screen.blit(asteroid, (x, y))


# text settings
def text_objects(text, font):
    textSurface = font.render(text, True, WHITE)
    return textSurface, textSurface.get_rect()


def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf', 60)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((screen_size[0] / 2), (screen_size[1] / 2))
    screen.blit(TextSurf, TextRect)
    pygame.display.update()
    time.sleep(2)


def crash_message_display(text, num):
    score = str(num)
    largeText = pygame.font.Font('freesansbold.ttf', 60)
    mediumText = pygame.font.Font('freesansbold.ttf', 40)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((screen_size[0] / 2), (screen_size[1] / 2))
    screen.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects("Score : "+score, mediumText)
    TextRect.center = ((screen_size[0] / 2), (screen_size[1] / 1.8))
    screen.blit(TextSurf, TextRect)
    pygame.display.update()
    time.sleep(2)
    game_start()


def score_display(count,high_score):
    font = pygame.font.SysFont(None, 25)
    text = font.render("Score: " + str(count), True, WHITE)
    text2 = font.render("High Score: " + str(high_score), True, WHITE)
    screen.blit(text2, (screen_size[0] * 0.75, 0))
    screen.blit(text, (screen_size[0] * 0.1, 0))


# Function to create buttons for home screen
def button(msg, x, y, w, h, flag):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(screen, BLUE, (x, y, w, h))
        if click[0] == 1:
            episode = 0
            high_score = 0
            q_matrix = np.zeros((5, 3))
            game_start(high_score, q_matrix, episode, flag)
    else:
        pygame.draw.rect(screen, DARK_BLUE, (x, y, w, h))

    smallText = pygame.font.Font("freesansbold.ttf",20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ((230 + (100 / 2)), (y + (h/2)))
    screen.blit(textSurf, textRect)


# Game home screen
def game_intro():
    intro = True

    while intro:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.fill(BLACK)
        largeText = pygame.font.Font('freesansbold.ttf', 90)
        TextSurf, TextRect = text_objects("SPACE RUN", largeText)
        TextRect.center = ((screen_size[0] / 2), 300)
        screen.blit(TextSurf, TextRect)

        # print(mouse)
        button("Human Player", 180, 450, 200, 50, "human")
        button("Computer Bot", 180, 550, 200, 50, "bot1")
        button("Computer Bot 2.0", 180, 650, 200, 50, "bot2")

        pygame.display.update()
        clock.tick(60)


# Q Learning algorithm
def game_bot(ship_x, asteroid_x, asteroid_size, ship_size, q_matrix, episode, player):

    lr = 0.00025
    dr = 0.001
    action_set = [-5, 0, 5]

    prev_state = get_state(ship_x, asteroid_x, asteroid_size, ship_size)

    action = take_action(prev_state, q_matrix, episode, player )

    cur_state, reward = new_state(action_set[action], ship_x, asteroid_x, asteroid_size, ship_size)
    # bellman equation
    if random.randrange(0, 2):
        q_matrix[prev_state, action] = q_matrix[prev_state, action] + lr * (reward + np.max(q_matrix[cur_state:]) - dr * q_matrix[prev_state, action])
        q_matrix[prev_state, action] = sigmoid(q_matrix[prev_state, action])

    return action_set[action]


# Returns new state and reward for the old state
def new_state(action, ship_x, asteroid_x, asteroid_size, ship_size):

    ship_x = ship_x + action

    if asteroid_x < ship_x < asteroid_size + asteroid_x:
        reward = -100
    elif asteroid_x < ship_x + ship_size < asteroid_size + asteroid_x:
        reward = -100
    elif ship_x > (screen_size[0] - ship_size - 25) or ship_x < 25:
        reward = -100
    else:
        reward = 1

    if action != 0:
        reward = -5

    return get_state(ship_x, asteroid_x, asteroid_size, ship_size), reward


# takes action depending on the state
def take_action(state, q, episode, player):

    if player == "bot2":
        b = np.load('qvalue.npy')
        action = np.argmax(b, axis=1)
        return action[state]
    else:

        if episode < 10:
            return random.randint(0, 2)
        else:
            action = np.argmax(q, axis=1)
            return action[state]


# returns the state of the agent relative to the obstacle
def get_state(ship_x, asteroid_x, asteroid_size, ship_size):

    if asteroid_x - 10 < ship_x < asteroid_size + asteroid_x + 20:
        if asteroid_x > 300:
            return 0
        else:
            return 1
    elif asteroid_x - 10 < ship_x + ship_size < asteroid_size + asteroid_x + 20:
        if asteroid_x < 100:
            return 2
        else:
            return 3
    else:
        return 4


def crash(ship_x, ship_y, size, radius, x, y):
    xx = x + radius
    yy = y + radius
    if ship_x > (screen_size[0] - size - 15) or ship_x < 15:
        return True

    for i in range(0, 360):
        cx = radius * math.cos(i) + xx
        cy = radius * math.sin(i) + yy
        if ship_y <= int(cy):
            if ship_x <= int(cx) <= ship_x + size:
                return True


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def game_start(high_score, q_matrix, episode, player):

    # ship position and size
    ship_size = 70
    ship_pos_x = 220
    ship_pos_y = 700
    x_change = 0

    # asteroid size and position
    asteroid_size = 120
    asteroid_pos_x = random.randrange(0, screen_size[0]-asteroid_size)
    asteroid_pos_y = -500
    asteroid_speed = 7
    asteroid_dodged = 0

    game_score = 0

    while True:

        for event in pygame.event.get():  # User action
            if event.type == pygame.QUIT:  # If user clicked close
                pygame.quit()
                quit()  # Flag that we are done so we exit this loop

            if player == "human":

                if event.type == pygame.KEYDOWN:  # movement of the ship
                    if event.key == K_a:
                        x_change = -5
                    elif event.key == K_d:
                        x_change = 5

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_a or event.key == pygame.K_d:
                        x_change = 0

            if event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    game_intro()

        if player != "human":
            x_change = game_bot(ship_pos_x, asteroid_pos_x, asteroid_size, ship_size, q_matrix, episode, player)

        ship_pos_x += x_change
        # Set the screen background
        screen.blit(background, (0, 0))
        score_display(game_score, high_score)
        # Draws images on the screen
        create_ship(ship_pos_x, ship_pos_y, ship_size)
        create_asteroid(asteroid_pos_x, asteroid_pos_y, asteroid_size)
        asteroid_pos_y += asteroid_speed

        # Show crash message if the ship collides on the border or asteroids
        if crash(ship_pos_x, ship_pos_y, ship_size, asteroid_size / 2, asteroid_pos_x, asteroid_pos_y):
            episode = episode + 1
            print("Episode : ", episode, "  Score : ", game_score)
            game_start(high_score, q_matrix, episode, player)

        if asteroid_pos_y > screen_size[1]:
            asteroid_size = random.randrange(12, 18) * 10
            asteroid_pos_y = 0 - asteroid_size
            asteroid_pos_x = random.randrange(0, screen_size[0] - asteroid_size)
            asteroid_dodged += 1
            game_score += 1
            if high_score < game_score:
                high_score = game_score
            score_display(game_score, high_score)
            if asteroid_dodged % 5 == 0:
                asteroid_speed += 1

        # Updates the whole screen
        pygame.display.flip()
        # FPS
        clock.tick(60)


game_intro()

