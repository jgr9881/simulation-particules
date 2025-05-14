import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.animation import FuncAnimation, FFMpegWriter

from config import *

class Poisson:
    def __init__(self, position, vitesse, couleur='blue'):
        self.position = np.array(position, dtype=float)
        self.vitesse = np.array(vitesse, dtype=float)
        self.couleur = couleur

    def update_position(self, dt, limite_zone):
        self.position += self.vitesse * dt
        for i in range(3):
            if self.position[i] < -limite_zone or self.position[i] > limite_zone:
                self.vitesse[i] = -self.vitesse[i]
                self.position[i] = np.clip(self.position[i], -limite_zone, limite_zone)

class BancPoissonsAoki:
    def __init__(self, num_poisson=NUM_FISH, limite_zone=ZONE_LIMIT):
        self.num_poisson = num_poisson
        self.limite_zone = limite_zone
        self.poissons = []
        for _ in range(num_poisson):
            pos = np.random.uniform(-limite_zone*BOX_PROPORTIONS,
                                     limite_zone*BOX_PROPORTIONS, size=3)
            vel_dir = np.random.uniform(-1, 1, size=3)
            vel_dir /= np.linalg.norm(vel_dir)
            vel = vel_dir * FISH_SPEED
            self.poissons.append(Poisson(pos, vel))

    def update(self, dt=DT):
        positions = np.array([f.position for f in self.poissons])
        tree = KDTree(positions)

        new_vitesses = []
        for i, poisson in enumerate(self.poissons):
            pos_i = poisson.position
            vel_i = poisson.vitesse

            idxs = tree.query_ball_point(pos_i, r=R_ATTRACTION)
            idxs = [j for j in idxs if j != i]

            F_rep = np.zeros(3)
            F_ali = np.zeros(3)
            F_att = np.zeros(3)
            count_ali = 0

            for j in idxs:
                voisin = self.poissons[j]
                d_vec = voisin.position - pos_i
                dist = np.linalg.norm(d_vec)
                if dist < 1e-6:
                    continue
                if dist < R_REPULSION:
                    F_rep += -K_REPULSION * (d_vec / dist)
                elif dist < R_ALIGNMENT:
                    F_ali += voisin.vitesse
                    count_ali += 1
                else:
                    F_att += K_ATTRACTION * (d_vec / dist)

            if count_ali > 0:
                F_ali = F_ali / count_ali
            v_new = vel_i + F_rep + F_ali + F_att
            norm = np.linalg.norm(v_new)
            if norm > 1e-6:
                v_new = v_new / norm * FISH_SPEED
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

def visualiser_banc_aoki(banc, frames=1000, interval=50, save_mp4=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-banc.limite_zone, banc.limite_zone)
    ax.set_ylim(-banc.limite_zone, banc.limite_zone)
    ax.set_zlim(-banc.limite_zone, banc.limite_zone)
    ax.set_title('Banc de Poissons - Modèle de Aoki')

    scatter = ax.scatter([], [], [], s=MARKER_SIZE, alpha=0.7, c=FISH_COLOR)

    def init():
        scatter._offsets3d = ([], [], [])
        return scatter,

    def update(frame):
        banc.update()
        pos = banc.get_positions()
        scatter._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        return scatter,

    ani = FuncAnimation(fig, update, init_func=init,
                        frames=frames, interval=interval, blit=False)
    if save_mp4:
        writer = FFMpegWriter(fps=FPS)
        ani.save('aoki_banc.mp4', writer=writer)

    plt.show()

if __name__ == '__main__':
    banc = BancPoissonsAoki()
    visualiser_banc_aoki(banc, save_mp4=save_mp4, frames=FRAMES)
