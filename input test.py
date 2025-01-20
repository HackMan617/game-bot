import pygame as pg
from pygame.locals import *

# Initialize Pygame
pg.init()
screen = pg.display.set_mode((640, 480))
clock = pg.time.Clock()

def main():
    # Initial square properties
    ORIGINAL_SIZE = 20  # Constant for the original size
    square_size = ORIGINAL_SIZE
    
    # Box properties (the container that follows the mouse)
    BOX_SIZE = 50  # Size of the containing box
    box_thickness = 2  # Thickness of the box outline
    
    while True:
        # Fill screen with white background
        screen.fill((255, 255, 255))
        
        # Get current mouse position
        mouse_x, mouse_y = pg.mouse.get_pos()
        
        # Calculate box position (centered on mouse)
        box_x = mouse_x - BOX_SIZE // 2
        box_y = mouse_y - BOX_SIZE // 2
        
        # Calculate square position (centered in box)
        square_x = mouse_x - square_size // 2
        square_y = mouse_y - square_size // 2
        
        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                return
            elif event.type == MOUSEBUTTONDOWN:
                print("Jump!")
                # Double the square size when clicked
                square_size = ORIGINAL_SIZE * 2
            elif event.type == MOUSEBUTTONUP:
                # Return to original size when mouse is released
                square_size = ORIGINAL_SIZE
        
        # Draw the red square
        pg.draw.rect(screen, (255, 0, 0), 
                     (square_x, square_y, square_size, square_size))
        
        # Draw the red box outline around the mouse
        pg.draw.rect(screen, (255, 0, 0), 
                     (box_x, box_y, BOX_SIZE, BOX_SIZE), 
                     box_thickness)
        
        # Update the display
        pg.display.flip()
        clock.tick(60)

# Execute game
main()
