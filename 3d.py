# Simple 3D test with Python + OpenGL
# pip install pygame PyOpenGL
# Kevin Walker 02 Jan 2025, ported from an old Processing sketch

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# Global variables to track mouse state
mouseX = 0
mouseY = 0
mousePressed = False
width = 640
height = 360

def init():
    glEnable(GL_DEPTH_TEST)
    # Remove lighting setup since we want pure white lines
    # glEnable(GL_LIGHTING)
    # glEnable(GL_LIGHT0)
    # glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 1, 0))
    # glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    # glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
    glColor3f(1.0, 1.0, 1.0)

def draw_box(size):
    glPushMatrix()
    glDisable(GL_LIGHTING)  # Disable lighting while drawing lines
    glScalef(size, size, size)
    vertices = (
        (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
        (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
    )
    edges = (
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,7), (7,6), (6,4),
        (0,4), (1,5), (2,7), (3,6)
    )
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()
    glEnable(GL_LIGHTING)  # Re-enable lighting if needed for other objects
    glPopMatrix()

def main():
    global mouseX, mouseY, mousePressed
    
    pygame.init()
    display = (width, height)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    init()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEMOTION:
                mouseX, mouseY = event.pos
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mousePressed = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mousePressed = False
                
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        cameraY = height/2.0
        fov = (mouseX/float(width)) * math.pi/2
        # Prevent division by zero
        fov = max(0.1, min(fov, math.pi/2 - 0.1))
        cameraZ = cameraY / math.tan(fov / 2.0)
        aspect = float(width)/float(height)
        
        if mousePressed:
            aspect = aspect / 2.0
            
        gluPerspective(45, aspect, 0.1, 1000.0)
        
        # Set up modelview
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -400)  # Move back to see the boxes
        
        # Apply rotations
        glRotatef(mouseY/float(height) * 180 - 90, 1, 0, 0)  # X rotation from vertical mouse
        glRotatef(mouseX/float(width) * 360, 0, 1, 0)  # Y rotation from horizontal mouse
        
        # Draw boxes
        draw_box(50)  # First box
        glTranslatef(0, 0, -50)  # Move back for second box
        draw_box(25)  # Second box
        
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main() 
