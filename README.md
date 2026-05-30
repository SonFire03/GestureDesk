# GestureDesk
[![Release](https://img.shields.io/badge/release-v1.0.0-00c2ff)](https://github.com/SonFire03/GestureDesk/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-2ea44f)](#installation)
[![License](https://img.shields.io/badge/license-MIT-7a42f4)](#)

Projet local Python/Linux pour piloter des actions PC non destructives via gestes de la main (camera USB, `/dev/video0`). Support 2 mains: affichage des 2 mains, controle par la main dominante (configurable).

## Description

GestureDesk est une application locale de hand-tracking pour Ubuntu qui transforme des gestes de la main en actions bureautiques utiles:
- deplacement du curseur a l'index,
- drag de fenetres,
- clic media,
- profils precision/performance.
- controle corps (MediaPipe Pose) pour volume/media.

Le projet est concu pour rester testable, modulaire et securise:
- mode `ARMED` active manuellement,
- actions non destructives uniquement,
- calibration sauvegardee localement,
- logs et tests unitaires inclus.

## Quick Start

```bash
git clone https://github.com/SonFire03/GestureDesk.git
cd GestureDesk
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
mkdir -p models
# Place ici models/hand_landmarker.task
./gestu
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
mkdir -p models
```

Puis placer un modele local `hand_landmarker.task` dans `models/`.

## Lancement

```bash
python run.py
```

Pre-check (sans lancer la fenetre):
```bash
python run.py --check
```

Config personnalisee:
```bash
python run.py --config config.json
```

Raccourcis clavier pendant l'execution:
- `q`: quitter proprement
- `a`: toggle manuel `armed/disarmed`
- `d`: forcer `disarmed`
- `u`: basculer UI `pro` / `debug`
- `o`: ouvrir/fermer UI Studio (reglages live)
- `j` / `k`: descendre/monter dans la liste Studio
- `h` / `l`: diminuer/augmenter la valeur selectionnee (sauvegarde immediate)
- `p`: activer/desactiver la trace du doigt (draw path)
- `f`: activer/desactiver le fallback HSV (tracking couleur bleu/vert)
- `v`: afficher/masquer la fenetre `GestureDesk Skeleton`
- `7`: mode mains uniquement (desactive body control)
- `8`: charger `pose_landmarker_lite.task`
- `9`: charger `pose_landmarker_full.task`

Affichage:
- Fenetre principale: `GestureDesk`
- Fenetre secondaire squelette: `GestureDesk Skeleton`

- <img width="1404" height="938" alt="image" src="https://github.com/user-attachments/assets/cb615076-28a6-4943-8ac0-f1c23ea80c9e" />
<img width="800" height="610" alt="video_2026-05-30_18-58-26-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/1663149f-ada8-4997-b448-7b6215f5a4c3" />


## Depannage camera `/dev/video0`

1. Verifier la camera:
```bash
ls -l /dev/video0
v4l2-ctl --list-devices
```
2. Si indisponible, verifier les permissions utilisateur (`video` group) et redemarrer la session.
3. L'application quitte proprement avec message clair si `camera_id` n'est pas accessible.

## Depannage Wayland / Xorg (PyAutoGUI)

Sous Wayland, `PyAutoGUI` peut ne pas fonctionner selon la configuration.
Si les actions souris/clavier ne repondent pas:
1. Se deconnecter.
2. A l'ecran de connexion, choisir la session **"Ubuntu on Xorg"**.
3. Reconnecter et relancer l'application.

## Depannage MediaPipe (erreur `API Hands`)

Ce projet utilise MediaPipe Tasks. Si tu vois:
`Modele MediaPipe introuvable: models/hand_landmarker.task`

alors ajoute un modele local:
1. Creer le dossier `models/` si absent.
2. Copier `hand_landmarker.task` dans `models/`.
3. Verifier `model_path` dans `config.json`.

Extrait de config:
```json
{
  "camera_id": 0,
  "model_path": "models/hand_landmarker.task",
  "mouse_smoothing_alpha": 0.35,
  "scroll_step": 120,
  "click_cooldown_seconds": 0.35
}
```

## Calibration (1 puis 2)

Reglages utiles dans `config.json`:
- `mouse_smoothing_alpha`:
  - plus faible (`0.2`) = curseur plus stable mais moins reactif
  - plus fort (`0.6`) = curseur plus reactif mais plus nerveux
- `mouse_adaptive_gain_min` / `mouse_adaptive_gain_max`:
  - accelere automatiquement les grands mouvements sans perdre la precision des petits
- `mouse_adaptive_scale_px`:
  - seuil de transition vers l'acceleration (plus grand = acceleration plus progressive)
- `drag_toggle_hold_seconds`:
  - maintien minimum du geste `deux doigts` pour activer/desactiver le drag (evite les faux positifs)
- `pinch_click_hold_seconds`:
  - maintien minimum du `pinch` avant clic (anti faux clic)
- `calibration_stage_seconds`:
  - duree de chaque etape du wizard calibration (centre + coins)
- `ui_mode`:
  - `pro` (compact) ou `debug` (details perf/etat)
- `dominant_hand_mode`:
  - `auto` = main dominante par taille apparente
  - `left` = force la main gauche (vue ecran)
  - `right` = force la main droite (vue ecran)
- `camera_width` / `camera_height`:
  - resolution de capture (plus bas = plus de FPS)
- `camera_fps`:
  - fps cible demande a la webcam (selon support reel du driver)
- `camera_fourcc`:
  - `MJPG` recommande pour une meilleure qualite/latence
- `camera_autofocus` / `camera_auto_exposure`:
  - active les reglages auto de la webcam
- `camera_exposure`:
  - valeur manuelle si `camera_auto_exposure=false`
- `inference_scale`:
  - redimensionnement applique uniquement a l'inference MediaPipe (0.5 a 1.0)
- `inference_every_n_frames`:
  - `1` = inference chaque frame (precision max)
  - `2` = inference une frame sur deux (FPS plus haut)
- `pose_inference_scale`:
  - redimensionnement dedie a l'inference corps (plus bas = plus rapide)
- `pose_inference_every_n_frames`:
  - cadence inference corps (`3` ou `4` conseille pour gagner des FPS)
- `ui_minimal`:
  - `true` pour desactiver le rendu visuel couteux de la main (FPS max)
- `draw_secondary_hand`:
  - `false` pour ne pas dessiner la main secondaire (gain FPS)
- `draw_finger_card`:
  - `false` pour retirer la carte doigts (gain FPS leger)
- `draw_path`:
  - active la trainée visuelle du doigt index
- `enable_hsv_fallback`:
  - fallback tracking couleur quand la main n'est pas detectee (utile demo)
- `enable_body_control`:
  - active le controle corps (pose epaules/poignets)
- `draw_pose_overlay`:
  - affiche le squelette corps complet
- `body_hold_seconds`:
  - maintien requis pour `deux mains levees` (anti faux positifs)
- `scroll_step`:
  - plus bas (`60`) = scroll plus doux
  - plus haut (`180`) = scroll plus rapide
- `click_cooldown_seconds`:
  - plus bas (`0.2`) = clic plus frequent
  - plus haut (`0.5`) = evite mieux les doubles clics involontaires

## Gestes disponibles (MVP)

- `poing ferme` -> idle/securite
- `a` au clavier -> toggle `armed/disarmed` (seulement manuel)
- `index leve` (main dominante) -> deplacement souris (si armed)
- `deux doigts (index+majeur)` (main dominante) -> activer/desactiver drag souris (si armed)
- `pincement pouce+index` -> clic gauche (si armed, avec cooldown)
- `main ouverte rapide` -> play/pause media (si armed, avec cooldown)
- `main droite levee (corps)` -> volume +
- `main gauche levee (corps)` -> volume -
- `deux mains levees maintenues` -> play/pause media

Deplacer une fenetre:
1. Passer en `ARMED`.
2. Pointer la barre de titre avec `index leve`.
3. Faire `deux doigts` pour demarrer le drag.
4. Revenir a `index leve` et bouger la main pour deplacer.
5. Refaire `deux doigts` pour relacher.

## Securite

- Mode par defaut: `DISARMED`.
- Tant que `DISARMED`, aucun geste ne declenche d'action OS.
- Actions autorisees limitees a: souris, clic gauche, drag souris, media play/pause, volume +/-.
- Aucune action destructive n'est implementee (pas de fermeture d'app, suppression de fichier, arret machine, execution de commande shell).
- Cooldowns pour eviter les repetitions involontaires.
- Logs locaux dans `logs/gesturedesk.log`.

## Tests

```bash
pytest -q
```

Couvre la logique pure:
- distance
- doigts leves (donnees simulees)
- cooldown
- toggle armed/disarmed
- mapping geste -> action

## Preset FPS

Pour maximiser les FPS sur machine moyenne:
```json
{
  "camera_fps": 60,
  "camera_width": 960,
  "camera_height": 540,
  "inference_scale": 0.60,
  "inference_every_n_frames": 2,
  "ui_minimal": true,
  "draw_secondary_hand": false,
  "draw_finger_card": false
}
```


## Profils a chaud

Pendant l'execution:
- `c` = wizard calibration (centre + 4 coins)
- `1` = precision
- `2` = balanced
- `3` = performance

Le profil ajuste inference, lissage souris et zone active.

La calibration lancee avec `c` est sauvegardee automatiquement dans `config.json`.
