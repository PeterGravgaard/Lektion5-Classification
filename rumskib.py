import pygame
import random

# Initialiser pygame
pygame.init()

# Definer skærmens dimensioner
bredde = 800
højde = 600

# Opret skærmen
skærm = pygame.display.set_mode((bredde, højde))
pygame.display.set_caption("Rumskibsspil")

# Definer farver
sort = (0, 0, 0)
hvid = (255, 255, 255)
rød = (255, 0, 0)

# Indlæs rumskibsbillede
rumskib_billede = pygame.image.load("rumskib.png")
rumskib_bredde = 64
rumskib_højde = 64

# Definer rumskibets startposition
rumskib_x = bredde // 2 - rumskib_bredde // 2
rumskib_y = højde - rumskib_højde - 10
rumskib_hastighed = 10

# Fjendtlige objekter
fjende_bredde = 50
fjende_højde = 50
fjender = []
for i in range(5):
    fjender.append([random.randint(0, bredde - fjende_bredde), random.randint(-150, -50)])

fjende_hastighed = 5

# Opret clock objekt
clock = pygame.time.Clock()

# Main loop til spillet
def spil_loop():
    global rumskib_x
    score = 0
    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # Håndter tastetryk
        taster = pygame.key.get_pressed()
        if taster[pygame.K_LEFT] and rumskib_x > 0:
            rumskib_x -= rumskib_hastighed
        if taster[pygame.K_RIGHT] and rumskib_x < bredde - rumskib_bredde:
            rumskib_x += rumskib_hastighed

        # Opdater fjendernes position
        for fjende in fjender:
            fjende[1] += fjende_hastighed
            if fjende[1] > højde:
                fjende[0] = random.randint(0, bredde - fjende_bredde)
                fjende[1] = random.randint(-150, -50)
                score += 1

            # Tjek for kollision med rumskibet
            if rumskib_x < fjende[0] + fjende_bredde and rumskib_x + rumskib_bredde > fjende[0] and rumskib_y < fjende[1] + fjende_højde and rumskib_y + rumskib_højde > fjende[1]:
                game_over = True

        # Tegn spillet
        skærm.fill(sort)
        skærm.blit(rumskib_billede, (rumskib_x, rumskib_y))
        for fjende in fjender:
            pygame.draw.rect(skærm, rød, (fjende[0], fjende[1], fjende_bredde, fjende_højde))

        # Vis score
        skrift = pygame.font.SysFont("bahnschrift", 25)
        score_tekst = skrift.render("Score: " + str(score), True, hvid)
        skærm.blit(score_tekst, [10, 10])

        pygame.display.update()

        # Sæt opdateringshastigheden
        clock.tick(60)

    pygame.quit()
    quit()

spil_loop()