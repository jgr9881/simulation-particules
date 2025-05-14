import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.animation import FuncAnimation, FFMpegWriter

from config import *  # Importation des paramètres si config.py existe

class Poisson:
    def __init__(self, position, vitesse, couleur='blue'):
        self.position = np.array(position, dtype=float)
        self.vitesse = np.array(vitesse, dtype=float)
        self.couleur = couleur
        self.voisins_visibles = []

    def update_position(self, dt, limite_zone):
        self.position += self.vitesse * dt
        for i in range(3):
            if self.position[i] < -limite_zone or self.position[i] > limite_zone:
                self.vitesse[i] = -self.vitesse[i]
                self.position[i] = np.clip(self.position[i], -limite_zone, limite_zone)

class BancPoissonsReseauInfluence:
    def __init__(self, nb_poissons=NUM_FISH, limite_zone=ZONE_LIMIT, vision_angle_deg=VISION_ANGLE):
        self.nb_poissons = nb_poissons
        self.limite_zone = limite_zone
        self.vision_angle = np.deg2rad(vision_angle_deg) / 2  # half-angle
        self.poissons = []
        for _ in range(nb_poissons):
            pos = np.random.uniform(-limite_zone*BOX_PROPORTIONS,
                                     limite_zone*BOX_PROPORTIONS, size=3)
            vel_dir = np.random.uniform(-1, 1, size=3)
            vel_dir /= np.linalg.norm(vel_dir)
            vel = vel_dir * FISH_SPEED
            self.poissons.append(Poisson(pos, vel))

    def update(self, dt=DT):
        positions = np.array([f.position for f in self.poissons])
        directions = np.array([f.vitesse / np.linalg.norm(f.vitesse) for f in self.poissons])

        new_vitesses = []
        
        for poisson in self.poissons:
            poisson.voisins_visibles = []

        for i, poisson in enumerate(self.poissons):
            pos_i = poisson.position
            dir_i = directions[i]
            vel_i = poisson.vitesse

            F_rep = np.zeros(3)
            F_ali = np.zeros(3)
            F_att = np.zeros(3)
            count_ali = 0

            for j, voisin in enumerate(self.poissons):
                if j == i:
                    continue
                d_vec = voisin.position - pos_i
                dist = np.linalg.norm(d_vec)
                if dist < 1e-6:
                    continue
                direction_voisin = d_vec / dist
                angle = np.arccos(np.clip(np.dot(dir_i, direction_voisin), -1.0, 1.0))
                
                if angle <= self.vision_angle:
                    poisson.voisins_visibles.append(j)
                    
                    if dist < R_REPULSION:
                        F_rep += -K_REPULSION * direction_voisin
                    elif dist < R_ALIGNMENT:
                        F_ali += voisin.vitesse
                        count_ali += 1
                    elif dist < R_ATTRACTION:
                        F_att += K_ATTRACTION * direction_voisin

            if count_ali > 0:
                F_ali /= count_ali

            v_new = vel_i + F_rep + F_ali + F_att
            norm = np.linalg.norm(v_new)
            if norm > 1e-6:
                v_new = (v_new / norm) * FISH_SPEED
            else:
                v_new = vel_i

            new_vitesses.append(v_new)

        for poisson, v_new in zip(self.poissons, new_vitesses):
            poisson.vitesse = v_new
            poisson.update_position(dt, self.limite_zone)

    def get_positions(self):
        return np.array([f.position for f in self.poissons])

    def get_velocities(self):
        return np.array([f.vitesse for f in self.poissons])
    
    def get_visibility_links(self):
        liens = []
        for i, poisson in enumerate(self.poissons):
            for j in poisson.voisins_visibles:
                liens.append((poisson.position, self.poissons[j].position))
        return liens


def visualiser_reseau_influence(banc, frames=FRAMES, interval=50, save_mp4=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-banc.limite_zone, banc.limite_zone)
    ax.set_ylim(-banc.limite_zone, banc.limite_zone)
    ax.set_zlim(-banc.limite_zone, banc.limite_zone)
    ax.set_title('Réseau d\'influence - Banc de Poissons avec liens de visibilité directionnels')

    scatter = ax.scatter([], [], [], s=MARKER_SIZE, alpha=0.8, c=FISH_COLOR)
    
    lines = []

    def init():
        scatter._offsets3d = ([], [], [])
        for line in lines:
            if line in ax.get_lines():
                line.remove()
        lines.clear()
        return [scatter]

    def update(frame):
        banc.update()
        pos = banc.get_positions()
        scatter._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        
        for line in lines:
            if line in ax.get_lines():
                line.remove()
        lines.clear()
        
        lien_vision = banc.get_visibility_links()
        for start_pos, end_pos in lien_vision:
            line, = ax.plot([start_pos[0], end_pos[0]],
                           [start_pos[1], end_pos[1]],
                           [start_pos[2], end_pos[2]],
                           'r-', alpha=0.1, linewidth=0.5)
            lines.append(line)
        
        return [scatter] + lines

    ani = FuncAnimation(fig, update, init_func=init,
                        frames=frames, interval=interval, blit=False)
    if save_mp4:
        writer = FFMpegWriter(fps=FPS)
        ani.save('reseau_influence.mp4', writer=writer)

    plt.show()

if __name__ == '__main__':
    banc = BancPoissonsReseauInfluence()
    visualiser_reseau_influence(banc, save_mp4=save_mp4, frames=FRAMES)