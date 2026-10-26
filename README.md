<img width="1200" height="750" alt="Original" src="https://github.com/user-attachments/assets/55b996fd-99e7-4d95-90f6-a5ee84477810" />
Original
<img width="3600" height="2250" alt="IA" src="https://github.com/user-attachments/assets/7fac8a37-5dda-420a-a474-d3f121c4715e" />
IA

















# TEXFORGE

Remaster IA des textures de jeux PC anciens (DirectX 9 et antérieurs), en traitement
**hors-ligne et 100 % local** : les textures sont extraites une fois, routées vers le
traitement adapté, vérifiées automatiquement, puis réinjectées au lancement du jeu.
L'IA générative tourne sur ta machine via ComfyUI (headless) ; plus aucun appel cloud
n'est nécessaire.

```
textures extraites ─► scan + routage ─┬─► UI  (HUD, polices, icônes, cartes) ─► Real-ESRGAN / Waifu2x ncnn ─┐
     (Special K)                      ├─► LOCAL (normal maps, données, verre) ─► Lanczos                  ├─► validation ─► revue
                                      ├─► COMFYUI (pierre, métal, bois, tissu, végétation) ─► SDXL + ControlNet Tile ─┘      │
                                      └─► KEEP (minuscules, tables de couleurs)        │                                    ▼
                                                                                         └─ échec : reprise ou repli    QC final (DA, raccord, alpha)
                                                                                                                         ├─ correctifs auto ou rejet
                                                                                                                         ▼
                                                                                                    DDS + mipmaps (texconv / intégré) ─► relecture ─► qc_report.json
```

**Hors périmètre :** la géométrie (seules les textures changent), le temps réel, les jeux
en ligne avec anti-triche, les jeux DX11/DX12.

---

## 1. Installation (Windows)

1. Python 3.11 ou plus récent ([python.org](https://www.python.org/downloads/), coche « Add to PATH »).
2. Dans le dossier de TEXFORGE :
   ```
   pip install -r requirements.txt
   ```
3. **texconv** (recommandé) : `texconv.exe` depuis github.com/microsoft/DirectXTex/releases, posé
   dans `tools\` de TEXFORGE ou indiqué dans `[qc] texconv_exe`. Sans lui, l'encodeur intégré
   (DXT1/DXT3/DXT5) prend le relais.
4. **Sans ComfyUI** : TEXFORGE passe tout seul en mode **Real-ESRGAN** (voir « Mode sans ComfyUI »).
   `texforge.bat install-realesrgan` (ou le bouton de l'interface) télécharge `realesrgan-ncnn-vulkan`
   (45 Mo, version officielle, empreinte SHA-256 vérifiée) dans `tools\realesrgan-ncnn-vulkan\`.
   **Avec ComfyUI** (remaster génératif) : installé et lancé (voir le guide « Workflows ComfyUI pour TEXFORGE ») avec :
   - un checkpoint SDXL (ou SD 1.5) et le ControlNet Tile correspondant, **ou** Z-Image Turbo
     (`z_image_turbo_bf16.safetensors` dans `diffusion_models`, `qwen_3_4b.safetensors` dans
     `text_encoders`, `ae.safetensors` dans `vae`, huggingface.co/Comfy-Org/z_image_turbo) : famille
     « Z-Image Turbo », choisie seule par l'interface quand ComfyUI n'a aucun checkpoint mais a Z-Image ;
   - avant la première texture, TEXFORGE compare les modèles du workflow à ceux de ComfyUI : un modèle
     par défaut absent est remplacé par un équivalent présent (autre checkpoint SDXL, autre ControlNet
     Tile SDXL, autre précision du même modèle), le workflow Tile+Canny est désactivé si son ControlNet
     Union manque, sinon le lancement s'arrête tout de suite en disant quel fichier manque, dans quel
     dossier `models\…` et où le trouver ;
   - les nœuds `ComfyUI-seamless-tiling` (raccord des textures répétables), **facultatifs** : s'ils
     manquent, TEXFORGE les retire du workflow et prolonge lui-même les bords en boucle avant l'envoi
     (même chose pour `ColorTransfer` : les couleurs sont recalées après coup). Un nœud indispensable
     absent (ControlNet, KSampler…) arrête le lancement tout de suite, avec la liste ;
   - lancement conseillé : `python main.py --listen 127.0.0.1 --port 8188`.
5. **Moteur UI rapide** : télécharge `realesrgan-ncnn-vulkan` (et facultativement
   `waifu2x-ncnn-vulkan`) depuis leurs pages GitHub officielles, décompresse-les, puis renseigne
   le chemin de l'exécutable dans `[ui_upscale]`. Sans eux, l'UI passe en Lanczos (moins net).
6. Vérification :
   ```
   texforge.bat selftest      (test complet, sans GPU ni réseau)
   texforge.bat doctor        (dépendances, moteurs UI, connexion ComfyUI)
   ```
7. Interface graphique : double-clic sur **`texforge_gui.bat`** (voir « Interface graphique »).

`texforge.bat <commande>` équivaut à `python -m texforge <commande>` et fonctionne depuis
n'importe quel dossier.

## Interface graphique

`texforge_gui.bat` (ou `pythonw app_gui.py`) ouvre l'application : dossier d'entrée, dossier de
sortie, réglages, puis **LANCER LE PIPELINE**. La chaîne complète s'exécute : scan et routage,
génération (UI en local, matières dans ComfyUI avec ControlNet Tile et raccord circulaire), contrôle
qualité, DDS avec mipmaps dans le dossier de sortie.

| Zone | Contenu |
|---|---|
| Modèle / checkpoint | moteur (ComfyUI ou simulation sans GPU), serveur + « Tester », famille SDXL / SD 1.5 / workflow personnalisé, checkpoint et LoRA lus sur le serveur, force de la LoRA |
| Photoréalisme / DA | denoise (0,20–0,60) et ControlNet Tile, « ajuster par matériau » (décale les préréglages au lieu de les remplacer, valeurs effectives affichées), ×2/×4, côté maximal, lot d'essai |
| DDS | cible (DDS Special K ou PNG RTX Remix), format (identique à l'original, auto BC1/BC3, BC1, BC3, BC7 Direct3D 11, BGRA8), encodeur, chemin de texconv |
| Options | tout par l'IA, IA par morceaux à la taille finale, routage HUD/UI, seamless tiling (+ forcer, déconseillé), mipmaps, QC SSIM/DA (+ correctifs), régénération après changement de réglages |
| Suivi | barre de progression réelle (scan 10 %, génération 60 %, QC et export 30 %), étape en cours, temps restant de l'étape, compteurs, console colorée filtrable et enregistrable |
| Résultats | dossier de sortie, rapport QC (tableau filtrable), galerie avant / après |
| Avant / Après (à droite) | original à gauche, résultat à droite (IA, essai IA écarté au QC ou agrandissement sans IA, en couleur), zoom à la molette et déplacement synchronisés, « Ajuster » / « 1:1 », « Suivre » pendant un lancement, filtre « IA seulement » ; bouton ◧ de l'en-tête pour le masquer |
| Mise à jour (coin haut droit) | pastille grise : rien à installer ; verte : version plus récente sur GitHub, bouton « Mettre à jour » (voir « Mises à jour ») |

### Mises à jour depuis GitHub

Au démarrage puis toutes les 2 minutes, TEXFORGE lit la version du dernier commit du dépôt
[MrGeek472/TEXTFORGE](https://github.com/MrGeek472/TEXTFORGE) (`__version__` de
`texforge/__init__.py`) et la compare à celle installée.
- **Pastille grise** (`● v0.5.4`) : rien à installer. Cas inclus : dépôt injoignable (l'infobulle
  donne la raison) et version installée plus récente que celle du dépôt (pas encore publiée).
- **Pastille verte** (`● 0.5.5 disponible`) + bouton **⬇ Mettre à jour** : l'infobulle donne le
  dernier message de commit. Le bouton télécharge l'archive de ce commit précis et la contrôle
  **avant d'écrire quoi que ce soit** :
  1. chemins sûrs ;
  2. version attendue ;
  3. compilation de chaque fichier Python ;
  4. présence de tous les modules du projet importés (un dépôt où il manque un fichier est refusé) ;
  5. installation des dépendances avec pip si `requirements.txt` a changé ;
  6. import d'essai de la nouvelle version dans un processus à part.

  Ensuite seulement, les fichiers qui diffèrent sont remplacés, après copie des originaux dans
  `%LOCALAPPDATA%\TEXFORGE\mises_a_jour\sauvegarde-…` (3 dernières gardées). Une erreur
  d'écriture remet tout en place. L'application propose ensuite de redémarrer (bouton
  **↻ Redémarrer** si tu refuses). Bouton grisé pendant un traitement.
- **Clic sur la pastille** : vérifier maintenant, ouvrir le dépôt, couper la vérification
  automatique, **revenir à la version précédente** (fichiers sauvegardés remis en place).
- **Pas de suppression** : les fichiers locaux absents du dépôt ne sont jamais supprimés.
- **Installation suivie par git** (dossier `.git`) : rien n'est écrasé, mets-la à jour avec
  `git pull`.
- **Pour publier une version** : envoie sur GitHub **tous** les fichiers de la version (dossiers
  compris), avec `__version__` augmenté. Sans augmentation de version, rien n'est proposé.
- **Envoi partiel refusé** : `texforge/release.json` liste l'empreinte de chaque fichier de la version.
  Si le dépôt mélange une version neuve et des fichiers anciens (seul `texforge/` envoyé, par
  exemple), la mise à jour est refusée avant toute écriture, avec la liste des fichiers à renvoyer.
  Le README et la licence ne sont pas contrôlés : tu peux les adapter sur GitHub.
- **Vérifications fréquentes** : requête conditionnelle à l'API GitHub (réponse « rien de changé »
  quand le dépôt n'a pas bougé). Si GitHub limite les requêtes, la version est lue directement sur
  la branche par défaut jusqu'à la levée de la limite.
- **En ligne de commande** : `texforge.bat update --check`, `texforge.bat update`,
  `texforge.bat update --rollback`.

### Tout par l'IA, et IA par morceaux

**Tout passer par l'IA (ComfyUI)** envoie aussi à ComfyUI la peau et les visages, le HUD, les
aplats, les dégradés, les petites textures et les formats allongés. Trois catégories restent hors
IA : les normal maps et cartes de données (l'IA produirait des vecteurs ou des masques faux, donc
un éclairage cassé en jeu), les polices (lettres déformées) et les textures de moins de 8 px.

**Rendu photoréaliste créatif** (menu « Rendu ») : pour inventer beaucoup de détail, jusqu'à
réinventer la matière (denoise jusqu'à 1). Trois différences avec le rendu Fidèle :
- le prompt décrit une photo (« photorealistic, ultra detailed… ») plus la zone **Prompt à donner**,
  au lieu du prompt « texture de jeu » du matériau
  détecté : sur une image, la détection se trompe (un dessin pris pour du tissu se couvre de fils) ;
- une **passe globale** d'abord : le modèle voit l'image entière à environ 1 mégapixel et la réinvente
  au denoise du curseur ; les morceaux n'affinent ensuite qu'à la moitié de ce denoise (0,25 à 0,45),
  sinon chacun invente son propre motif ;
- le contrôle qualité n'est plus qu'un garde-fou (structure, couleurs et palette très assouplies) et
  le verrou des couleurs passe de 0,6 à 0,25.
Pour une **texture de jeu**, l'éclairage inventé (reflets, ombres) est faux en jeu : le moteur éclaire
déjà la texture. Ce rendu convient surtout aux images, illustrations et captures.

**Prompt à donner** (zone de texte) : ce que doit produire l'IA, ajouté au prompt de chaque texture
du lancement. Un modèle de diffusion génère une image qui correspond à une description : « rusty
scratched metal, deep cracks » marche mieux que « ajoute de la rouille ». SDXL comprend surtout
l'anglais ; Z-Image (encodeur Qwen3) comprend aussi le français. Sous la zone, l'aperçu montre le
prompt réellement envoyé. Case **Prompt seul, sans ajouts** : seul ce texte part (plus de préfixe,
de prompt de matériau ni de suffixe). Une règle `[[rules]]` avec `hint` s'ajoute toujours en tête.

Bouton **✨ Détailler en anglais…** : un modèle de langage local réécrit ton texte (en français ou en
anglais) en un prompt anglais beaucoup plus détaillé. Il décrit les matières, le micro-détail, l'usure
et les nuances de couleur, sans ajouter d'objet, de texte ou de logo. Le résultat s'ouvre dans un
**aperçu** modifiable avec trois boutons :
- **Utiliser ce prompt** remplace la zone (Ctrl+Z dans la zone rend ton texte) ;
- **↻ Régénérer** propose une autre version ;
- **Annuler** ne change rien.

Les consignes données au modèle dépendent de deux réglages :
- **Rendu** : *Fidèle* garde un éclairage neutre, sans ombres ni reflets incrustés, pour que le jeu
  éclaire lui-même la texture ; *Photoréaliste* vise un rendu de photographie.
- **Famille** : 30 à 60 mots pour SDXL / SD 1.5, dont l'encodeur CLIP ne lit qu'environ 75 mots ;
  70 à 130 mots pour Z-Image.

Moteur, dans cet ordre :
1. ComfyUI lui-même (nœud `TextGenerate` des versions récentes) avec l'encodeur Qwen3-4B de Z-Image
   Turbo (`qwen_3_4b.safetensors`, déjà présent si la famille Z-Image fonctionne). Le premier appel
   charge le modèle : 10 à 30 s.
2. Sinon LM Studio (port 1234) ou Ollama (port 11434), lancés avec un modèle.

Rien ne quitte le PC.

**Denoise auto (maximum)** (case sous le curseur) : le curseur se place au maximum du rendu et se
verrouille ; la valeur choisie à la main revient quand on décoche.
- Fidèle : 0,60, le plus fort denoise que la validation accepte d'ordinaire sans voir la texture
  transformée. Si elle la juge trop éloignée de l'original, TEXFORGE redescend d'un palier (0,49, puis
  0,41). Avec la génération par morceaux, le palier se choisit sur un essai à 1 mégapixel (quelques
  secondes) avant de lancer les morceaux ; sans morceaux, chaque nouvel essai de la texture prend le
  palier suivant. Résultat : le plus de détail possible sans transformer la texture.
- Photoréaliste créatif : 1,00 pour la passe globale, puis 0,85 et 0,70 si le garde-fou échoue.

**Famille Z-Image Turbo** : Z-Image Turbo (6 milliards de paramètres, 8 pas) avec son ControlNet
Tile officiel, `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps.safetensors` (2,0 Go, licence
Apache-2.0, huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) dans
`models\model_patches`. La version « lite » tient dans 16 Go de VRAM avec le modèle ; la complète
(6,7 Go) dépasse. Le ControlNet Tile tient les formes : le denoise des préréglages est multiplié
par 1,3. Sans son fichier, TEXFORGE le retire du workflow (img2img doux, denoise ×0,75) et le dit ;
à ce réglage, l'IA ne fait presque que lisser, le résultat ressemble à l'original.

**Real-ESRGAN** sert de base à l'IA par morceaux : sans lui, les morceaux partent d'un
agrandissement Lanczos flou et l'IA ajoute peu. Un chemin réglé qui n'existe plus est remplacé par
celui que TEXFORGE trouve (`tools\`, PATH), avec une ligne dans la console ; s'il n'y en a aucun,
un avertissement s'affiche au lancement.

**IA par morceaux à la taille finale** : le modèle travaille à environ 1 mégapixel (1024×1024 en
SDXL, 512×512 en SD 1.5). En une seule passe, une texture de 1024 px gagne du détail à 1024 px, puis
son passage en 4096 n'est qu'un agrandissement classique ; une texture de 2048 en perd même. Avec
l'option, la texture est d'abord agrandie à sa taille finale (Real-ESRGAN s'il est installé, sinon
Lanczos), puis générée morceau par morceau à la résolution native du modèle, avec 128 px de
recouvrement fondu entre morceaux et une marge en boucle pour les textures répétables. Coût : une
texture de 2048 = 9 morceaux, une de 4096 = 25 morceaux de 1024² (SDXL, 28 pas à denoise 0,42),
soit plusieurs minutes par texture en 4096 ; le temps restant affiché pendant la génération donne
la vraie mesure sur ta carte.
Réglages équivalents dans `texforge.toml` : `[routing] force_ai = true`, `[comfyui] tiled = true`,
`tile_size`, `tile_overlap`.

**Durée totale estimée** (ligne bleue sous la barre de progression) : dès la fin de l'analyse,
TEXFORGE annonce la durée du lancement entier (génération, puis contrôle qualité et export), le temps
restant et l'heure de fin ; la console l'écrit aussi (`[ESTIMATION] ≈ 2 h 10 au total — génération
~1 h 55 : 480 passage(s) ComfyUI × ~14 s…`).
- Le calcul compte les **passages** ComfyUI comme le moteur les fera : 1 par texture en une passe ;
  par morceaux, 1 par morceau (4096² = 25, 2048² = 9, plus la marge des textures répétables), plus la
  passe globale du rendu photoréaliste ou l'essai du denoise automatique. Agrandissements locaux,
  Real-ESRGAN et export se comptent au mégapixel produit.
- Pendant la génération, l'estimation se recale sur la durée réelle des passages déjà faits et sur
  les essais en plus des textures refusées par la validation ; le temps restant se décompte chaque
  seconde entre deux recalages.
- À la fin, la durée réelle s'affiche à côté de l'estimation de départ, et les durées mesurées sont
  mémorisées (`timings.json` dans `%LOCALAPPDATA%\TEXFORGE`, une valeur par famille de modèles) : le
  lancement suivant part d'elles. Au tout premier lancement, la durée d'un passage est une valeur
  indicative (Z-Image Turbo avec ControlNet Tile 16 s, SDXL 12 s, SD 1.5 3,5 s), donc l'annonce de
  départ peut être fausse du simple au double ; elle se corrige dès les premiers passages.

### Mode sans ComfyUI : Real-ESRGAN réaliste

Moteur **Real-ESRGAN — sans ComfyUI** : choisi automatiquement quand ComfyUI ne répond pas (tant
qu'aucun moteur n'a été imposé dans la liste), proposé aussi au lancement si ComfyUI est éteint.
Aucune IA générative : la matière n'est pas réinventée, elle est agrandie au plus près du réel.

Real-ESRGAN seul a deux défauts sur des textures de jeu : il **invente des motifs** sur les
matières bruitées (quadrillages, fissures, « murs » dans de la pierre ou du gravier) et il **lisse
le grain fin** (aspect cireux). TEXFORGE corrige les deux :

1. deux rendus ×4 : `realesrgan-x4plus` (net) et `realesr-animevideov3` (prudent, n'invente rien) ;
2. **garde anti-invention** : là où x4plus produit beaucoup plus de détail que le modèle prudent,
   c'est une structure inventée, on passe progressivement au rendu prudent ;
3. **verrou des couleurs** : teintes et éclairage repris de l'original (la DA ne bouge pas) ;
4. **accord du détail** : le détail attendu dans les octaves créées par l'agrandissement est
   prolongé depuis le spectre de l'original ; un grain de matière comble le manque, les
   contours en excès sont adoucis (curseur **Micro-détail**, 1 = dose mesurée) ;
5. raccord des textures répétables (marges circulaires, grain périodique) et alpha d'origine.

Mesuré sur des textures dont la version haute résolution est connue (brique, gravier, herbe,
pierre, pelage… réduites ×4 puis remontées) : structure (SSIM) 0,49 → 0,67, écart de spectre
8,8 → 4,8 dB, écart de couleur (ΔE) 2,9 → 0,6 par rapport à Real-ESRGAN brut. Le contrôle
qualité final s'applique comme pour ComfyUI. Réglages fins : section `[esrgan]` de la configuration.

**Dossier d'entrée automatique** : au démarrage, TEXFORGE cherche les textures déjà extraites —
dumps Special K (`SK_Res\dump\textures`, injection locale ou profil global) et captures RTX Remix
(`rtx-remix\captures`) — dans les jeux Steam, Epic et GOG, les programmes installés, les dossiers
`X:\Games`, `X:\Jeux`, `X:\GOG Games`… de chaque disque et les dossiers déjà utilisés. Si l'entrée
est vide, il prend le dump le plus récent et met en sortie le `SK_Res\inject\textures` du même
jeu (une sortie choisie à la main n'est jamais remplacée). **🔍 Détecter** liste les autres jeux
trouvés.

**Dossier de départ choisi à la main** (Parcourir, ou chemin collé puis Entrée) : TEXFORGE cherche
le dossier des textures qu'il contient et le met à la place, dans cet ordre :

1. un dossier `SK_Res\inject` (la sortie de TEXFORGE) choisi par erreur → le dump du même jeu ;
2. un dump Special K ou une capture RTX Remix (ou l'un de leurs sous-dossiers) → gardé tel quel ;
3. le dossier d'un jeu → son `SK_Res\dump\textures` (jusqu'à 4 niveaux sous le dossier, à côté de
   l'exécutable) ou son `rtx-remix\captures` ;
4. un jeu injecté par SKIF → le dump de son **profil Special K global** (`Documents\My Mods\SpecialK\
   Profiles\<jeu>`), reconnu au nom du dossier ou de l'exécutable ;
5. un jeu aux textures en fichiers séparés → le sous-dossier qui les regroupe (un dossier qui pèse
   moins de 2 % des textures, lanceur ou captures d'écran, est laissé de côté ; un dossier
   d'interface voisin est gardé) ;
6. rien de lisible → le dossier reste tel quel et TEXFORGE dit pourquoi : Unity (`.assets`), Unreal
   (`.pak`), Source (`.vpk`), Bethesda (`.bsa`/`.ba2`)… rangent leurs textures dans des archives, il
   faut d'abord le dump Special K. Le lancement est alors refusé, avec la même explication.

La ligne sous le champ affiche le résultat (« ✓ 1 234 texture(s) — dump Special K ») et, pour un dump
Special K, la sortie passe sur le `SK_Res\inject\textures` du même jeu (une sortie choisie à la main
n'est jamais remplacée).

Bibliothèque ailleurs : variable `TEXFORGE_SEARCH=D:\MesJeux;E:\Retro`.
En ligne de commande : `python -m texforge detect` (liste) ou `detect --json`.

**Langue** : le menu 🌐 en haut à droite passe l'interface de **Français** à **English** et
inversement, sans redémarrer (réglages, taille de fenêtre et journal conservés). Le choix est
mémorisé ; au premier lancement, la langue du système est prise. Il couvre l'interface, la
console, le rapport QC et la galerie ; il se change entre deux traitements. En ligne de commande :
`python -m texforge --lang en …` ou la variable `TEXFORGE_LANG=en` (l'aide `-h` reste en français).

**Annuler** finit les textures en cours puis s'arrête proprement ; un second clic, **Forcer
l'arrêt**, coupe le traitement et vide la file ComfyUI. Relancer reprend où l'on s'était arrêté :
rien n'est refait, rien n'est perdu.

Le traitement tourne dans un processus séparé, supervisé par un QThread : l'interface reste fluide,
un plantage du traitement ne ferme pas la fenêtre. Chaque dossier d'entrée a son projet dans
`%LOCALAPPDATA%\TEXFORGE\projets\` (état, résultats, `texforge.toml` généré). Toutes les commandes
habituelles marchent dessus, et « Copier la commande » donne l'équivalent en ligne de commande :

```
python -m texforge remaster --settings "%LOCALAPPDATA%\TEXFORGE\projets\<projet>\gui_settings.json"
python -m texforge remaster --input C:\dump --output C:\Jeu\SK_Res\inject\textures --provider mock
```

## 2. Extraire les textures du jeu

### Jeux DX9 / DX8 / DX7 : Special K + dgVoodoo2

L'injection de textures de Special K est fiable en D3D11 ; en D3D9 elle n'est plus vraiment
maintenue. La méthode recommandée sur le forum Special K : **dgVoodoo2** convertit les appels
du jeu en D3D11, puis Special K extrait et injecte comme pour un jeu D3D11.

1. Installe dgVoodoo2 dans le dossier du jeu en suivant la page dgVoodoo du wiki Special K.
2. Lance le jeu avec Special K (SKIF ou injection locale).
3. Active l'extraction : dans le fichier de configuration Special K du jeu, section
   `[Textures.General]`, ajoute `DumpOnFirstLoad=true`.
4. Joue en traversant un maximum de zones : seules les textures affichées sont extraites.
5. Les fichiers arrivent dans `SK_Res\dump\textures\...`.

### Jeux DX8/DX9 à pipeline fixe (≈1998-2004) : RTX Remix

Fais une capture avec RTX Remix, puis pointe `inputs` vers le dossier des textures capturées.
Export avec `--target remix`.

### Fallout 4, Fallout 76, Starfield : archives BA2 (rien à extraire à la main)

Choisis le dossier du jeu (ou son `Data`) : TEXFORGE lit directement les archives de textures
`.ba2` (type DX10, versions 1, 2, 3, 7 et 8 : Fallout 4 d'origine et « next-gen », Starfield).
Fallout 4 et Fallout 76 sont compressés en zlib ; Starfield en LZ4 (module `lz4`, installé avec
`requirements.txt`).

- **Filtre** (champ « Archives BA2 », `--ba2-filter`) : `*_d.dds` par défaut, les textures de
  couleur. Les cartes `_n` (relief, BC5) et `_s` (spéculaire, BC5) sont des données : agrandies sans
  IA, et réécrites en BGRA8 non compressé faute d'encodeur BC5, donc 4 fois plus lourdes. Un motif
  avec `/` porte sur le chemin : `textures/architecture/*_d.dds;textures/landscape/*_d.dds`.
- **Ce qui est retenu** : seulement ce que l'agrandissement change (une texture déjà à la taille
  maximale est laissée telle quelle) ; une texture remplacée par un fichier séparé dans `Data`
  (mod) est ignorée, comme le fait le jeu ; les cubemaps sont ignorées. Si deux archives contiennent
  le même chemin, la dernière dans l'ordre alphabétique l'emporte (archives du jeu d'abord), ce qui
  approche l'ordre de chargement sans le garantir.
- **Lot d'essai** : il s'applique dès l'extraction, sur un échantillon réparti dans tout le jeu
  (toujours le même), pour un essai rapide.
- Les textures retenues sont extraites dans le dossier de projet (`…\TEXFORGE\projets\<jeu>\ba2`),
  puis traitées comme les autres. Une extraction est reprise là où elle s'est arrêtée.
- **Sortie** : cible « DDS — fichiers séparés » (`--target loose`), même chemin que dans le jeu
  (`Textures\Architecture\…_d.DDS`), dans `<jeu>\TEXFORGE_Remaster` par défaut. TEXFORGE n'écrit
  jamais dans `Data`. Pour installer : copie le dossier `Textures` de `TEXFORGE_Remaster` dans `Data`,
  ou installe `TEXFORGE_Remaster` comme un mod (Mod Organizer 2, Vortex).
- Pour que le jeu préfère ces fichiers aux archives : `Documents\My Games\Fallout4\Fallout4Custom.ini`
  (ou `StarfieldCustom.ini`), section `[Archive]` :

  ```ini
  [Archive]
  bInvalidateOlderFiles=1
  sResourceDataDirsFinal=
  ```
- Ordre de grandeur sur un Fallout 4 complet : environ 8 400 textures `_d` (11 Go en archives),
  surtout en 1024 et 2048 px. Tout traiter prend des heures, et la sortie ×2 pèse environ 4 fois
  plus : commence par un dossier (`textures/architecture/*_d.dds`) ou un lot d'essai.
- En ligne de commande : `python -m texforge ba2 list "<jeu>"` (textures retenues par le filtre),
  `python -m texforge ba2 extract "<jeu>" <dossier>` ; `remaster --input "<jeu>"` passe tout seul en
  mode BA2.
- Non fait : réécrire une archive `.ba2` (il faudrait aussi un plugin `.esp` pour la charger) ; les
  fichiers séparés suffisent avec les deux lignes ci-dessus.

### Jeux Unreal Engine (3, 4, 5)

TEXFORGE reconnaît un jeu Unreal quand tu choisis son dossier, en lisant seulement les en-têtes de
ses archives (`.pak`, conteneurs IoStore `.utoc/.ucas`, paquets `.upk` d'Unreal 3) : version,
chiffrement, compression, signature. Il ne déchiffre ni ne décompresse rien.

- **Archives chiffrées (AES)**, le cas de la plupart des jeux récents : TEXFORGE ne les ouvre pas,
  ce serait contourner la protection du jeu.
- **Archives non chiffrées** : les textures sont dans des fichiers `.uasset` qu'Unreal sérialise
  selon sa version ; TEXFORGE ne les lit pas (pas encore).
- **La voie qui marche** : la capture en jeu avec Special K, qui ne remplace les textures qu'en
  Direct3D 11 (le Direct3D 12 n'est pas géré par ses mods de textures) :
  1. Special K pour le jeu (SKIF, ou `dxgi.dll` dans `<Projet>\Binaries\Win64`) ;
  2. option de lancement Steam `-dx11` (Propriétés du jeu → Options de lancement) ;
  3. dump des textures, partie, puis choisir à nouveau le dossier du jeu : TEXFORGE trouve
     `Binaries\Win64\SK_Res\dump\textures` et met la sortie sur `SK_Res\inject\textures`.
- Limites : un jeu Unreal 5 qui impose Direct3D 12 (Nanite, Lumen) refuse parfois `-dx11` ; Unreal
  charge ses textures par morceaux pendant la partie (streaming), si bien que la capture peut être
  partielle. À vérifier jeu par jeu.
- Unreal 3 (souvent Direct3D 9) : même voie, avec dgVoodoo2 d'abord (section précédente).

## 3. Configurer

```
texforge.bat init C:\Remaster\MonJeu
```
Dans `C:\Remaster\MonJeu\texforge.toml`, règle au minimum :
```toml
[paths]
inputs = ['C:\Jeux\MonJeu\SK_Res\dump\textures']

[comfyui]
server = "http://127.0.0.1:8188"
workflow = "workflows/texforge_sdxl_tile_api.json"
workflow_edges = "workflows/texforge_sdxl_tile_canny_api.json"

[ui_upscale]
realesrgan_exe = 'C:\Outils\realesrgan-ncnn-vulkan\realesrgan-ncnn-vulkan.exe'
```
Tout le reste a des valeurs par défaut commentées dans le fichier.

## 4. Routage

`scan` mesure chaque texture (sur une réduction ≤ 256 px, insensible aux blocs DXT) et lui
attribue une route :

| Route | Critères | Traitement |
|---|---|---|
| **UI** | nom explicite (`hud`, `font`, `icon`, `minimap`, `menu`, `button`…) ou score UI ≥ `ui_threshold` : aplats, peu de couleurs, contours nets, transparence découpée, glyphes | Real-ESRGAN `realesr-animevideov3` → Waifu2x → Lanczos, par dossier entier (1 lancement par lot) |
| **COMFYUI** | textures de surface avec du détail : matériau détecté (pierre, maçonnerie, métal, bois, tissu, végétation, générique) ; avec `force_ai = true`, tout sauf normal maps, cartes de données, polices et textures minuscules | workflow Tile (ou Tile+Canny pour les joints), denoise / force ControlNet par matériau, par morceaux si `tiled = true` |
| **LOCAL** | normal maps (Z dans le bleu ; DXT5nm d'Unity et d'autres moteurs, X dans l'alpha et Y dans le vert ; BC5, X et Y dans le rouge et le vert — reconnues au contenu, même avec un nom en empreinte), masques et cartes de données, verre, textures trop petites ou trop allongées ; peau si `skin_route = "local"` | Lanczos canal par canal, jamais d'IA |
| **KEEP** | < 8 px, tables de couleurs | copie |

Les noms Special K (`Compressed_1A2B3C4D.dds`) sont des hachages : sur un vrai dump, le routage
repose donc presque entièrement sur l'analyse d'image. **Contrôle-le avant le premier `run`** :

```
texforge.bat route                        (résumé par route et par matériau)
texforge.bat route --report routage.csv   (détail : score UI, matériau, raison, mesures)
```

Corrections : `ui_threshold` (plus bas = plus d'UI) ou une règle par texture, prioritaire sur tout :

```toml
[[rules]]
match = "*Compressed_1A2B3C4D*"   # motif sur le nom de fichier
mode = "ui"                       # "ai" (ComfyUI) | "ui" | "upscale" (Lanczos) | "keep"
engine = "nearest"                # UI seulement : realesrgan, waifu2x, nearest (pixel art), lanczos
# material = "masonry"            # ComfyUI : stone, masonry, metal, wood, fabric, vegetation, skin, generic
# hint = "mossy cobblestone"      # précision ajoutée au prompt
# tile = true                     # force « texture répétable »
```

## 5. Utiliser

| Étape | Commande | Effet |
|---|---|---|
| Analyse | `texforge.bat scan` | Analyse et routage |
| Contrôle | `texforge.bat route --report routage.csv` | Vérifier le routage |
| Essai sans GPU | `texforge.bat run --provider mock` | Toute la chaîne sans ComfyUI |
| UI seule | `texforge.bat run --stage ui` | Étape UI (secondes) |
| Premier lot IA | `texforge.bat run --stage ai --limit 20` | 20 textures ComfyUI |
| Revue | `texforge.bat review` | Galerie avant/après |
| Tout | `texforge.bat run` | UI, puis local, puis ComfyUI |
| Échecs | `texforge.bat failures` | Journal des échecs par type |
| Reprise | `texforge.bat retry --kind oom` puis `run` | Remet en file les échecs choisis |
| Arrêt propre | `texforge.bat stop` (autre console) | Termine les textures en vol puis s'arrête |
| Contrôle qualité | `texforge.bat qc` | Mesure, corrige ou rejette chaque résultat (sans export) |
| Export | `texforge.bat export --target specialk --out C:\Jeux\MonJeu\SK_Res\inject\textures` | QC + DDS avec mipmaps + `qc_report.json` |
| Hors projet | `texforge.bat postprocess --originals dump --generated ComfyUI\output --out inject` | Même chose sur un dossier de sorties ComfyUI |

Puis dans Special K : `Inject=true` dans la section `[Textures.D3D11]`, relance le jeu.

### Tableau de bord

```
── Étape UI : 5 texture(s)
[UI] realesrgan (realesr-animevideov3, ×4) : 5 image(s) en 0.3 s
[UI] HUD_01 traité localement (realesrgan, ×4) (3/5)
── Étape COMFYUI : 7 texture(s)
[COMFYUI] groupe : workflow Tile, raccord enable, 3 texture(s)
[COMFYUI] Wall_04 en file (3/45) — masonry, tiling enable
[RETRY] Rock_02 : [oom] torch.OutOfMemoryError: CUDA out of memory … — nouvel essai 1
[GPU] VRAM libérée (mémoire saturée (OOM) avant nouvel essai) — libre avant : 1.0 Go
[COMFYUI] Wall_04 validé — structure 0.84, essai 1, 7.9 s
[REPLI] Moss_07 → realesrgan (raccord 2.31 > 1.80)
== Terminé en 412 s — 120 UI, 30 local, 45 ComfyUI validées, 2 replis, 1 reprises, 0 échecs
```

Chaque événement est aussi écrit dans `texforge_work/events.jsonl` (une ligne JSON horodatée).

### GPU et mémoire

- Les textures ComfyUI sont traitées par lots (`batch_size`, défaut 12) avec au plus
  `max_in_flight` prompts en file côté serveur : la file ComfyUI ne gonfle jamais.
- Elles sont groupées par workflow (Tile puis Tile+Canny) et par mode de raccord, ce qui évite
  de recharger les modèles à chaque texture.
- Entre deux lots, `/free` décharge les modèles si la VRAM libre passe sous `vram_threshold_gb`
  (`free_vram = "auto"`), à chaque lot (`"always"`), ou toutes les `free_every` textures.
- OOM : libération immédiate, puis nouvel essai (`max_transient_retries`), puis repli Real-ESRGAN.
- Serveur coupé : attente du redémarrage (`server_wait`), puis repli ou arrêt (`on_server_down`).
- `max_consecutive_failures` arrête l'étape si tout échoue (workflow cassé, modèle manquant).

### Traçabilité des échecs

Chaque échec est enregistré dans la base (`texforge_work/state.db`, table `failures`) avec
l'étape, la route, le type (`oom`, `timeout`, `server`, `workflow`, `cuda`, `validation`,
`engine`, `image`, `error`), le message, le détail et l'action prise (`retry`, `fallback`, `failed`).

```
texforge.bat failures --kind oom --last 50
texforge.bat failures --json                       (pour un script)
texforge.bat retry --kind timeout                  (remet en file ces textures)
texforge.bat retry --route comfyui --include-fallback   (refait aussi les replis)
```

### Validation automatique

Chaque résultat est remis exactement à la géométrie de l'original (recadrage, mise à
l'échelle, raccord des textures répétables, transparence d'origine), puis comparé à l'original :

| Contrôle | Réglage | Défaut |
|---|---|---|
| Disposition générale conservée | `min_structure` | 0,65 |
| Pas de décalage global | `max_shift` | 3 % |
| Pas d'objet ajouté ou effacé (pire bloc 6×6) | `min_local` / `max_local_color` | 0,3 / 0,12 |
| Raccord invisible en répétition | `max_seam_ratio` | 1,8 |

Échec → nouvel essai avec une autre graine (jusqu'à `max_attempts`), puis repli Real-ESRGAN.
**Ces seuils et ceux du routage ont été calibrés sur des textures synthétiques**, pas encore
sur de vraies sorties SDXL : après le premier lot de 20, regarde les scores dans la galerie et
ajuste `[quality]` et `[materials.*]`.

### Contrôle qualité final et DDS

`texforge.bat export` (ou `qc` seul) reprend chaque texture terminée et la soumet au QC final.
Principe : la direction artistique d'une texture tient dans ce que l'original contient **à sa
propre résolution** ; le micro-détail photoréaliste vit au-dessus. Les mesures de fidélité se font
donc sur le rendu **ramené à la taille d'origine**, les mesures de détail à pleine résolution.

| Famille | Mesures (réglages `[qc]`) | Échec → |
|---|---|---|
| Structure | corrélation, décalage, pire bloc (`[quality]`), SSIM multi-échelle `min_ssim` | rejet (version locale conservée) |
| Couleur / palette | ΔE2000 moyen et 95e centile, teinte, saturation, luminosité, distance d'histogramme Lab | verrou couleur, sinon rejet |
| Raccord (textures répétables) | cassure vs transitions courantes, vs colonnes voisines, dérive de couleur d'un bord à l'autre, motifs décalés | fondu du raccord / verrou couleur ; motifs décalés : rejet |
| Transparence | masque réinjecté depuis l'original, contrôle IoU (découpes) ou écart moyen (dégradés) | rejet |
| Détail | gain par rapport à un simple agrandissement, niveau de grain | avertissement |

- **Verrou couleur** : on ajoute au rendu l'écart (original − rendu réduit), flouté puis agrandi.
  Le rendu réduit retrouve exactement les couleurs et les valeurs de l'original ; le détail ajouté
  par l'IA au-dessus de la résolution d'origine n'est pas touché.
- **Fondu du raccord** : appliqué au seul détail de l'IA (rendu − original agrandi en boucle), donc
  un joint de carrelage ou une arête qui tombe sur le bord reste net.
- **Joint au bord** : quand l'original a un joint pile sur son bord (carrelage, mortier) et que le
  rendu, ramené à la résolution d'origine, reproduit ce saut au moins aussi fidèlement que
  l'original (`seam_joint_match`), les tests de cassure ne s'appliquent pas : un rendu net fait
  ressortir ce joint par rapport aux transitions internes sans qu'il y ait de cassure. La dérive de
  couleur et les motifs décalés restent contrôlés.
- Un choix fait à la main dans la galerie (`apply-review`) est **mesuré, jamais modifié**.
- Un rejet passe la texture en repli (statut `fallback`, échec de type `qc`) :
  `texforge.bat retry --kind qc --include-fallback` la renvoie en génération.
- Les résultats déjà contrôlés avec les mêmes réglages ne sont pas recalculés.

**DDS** : chaîne de mipmaps complète (`[output] mipmaps = "always"`), calculée par TEXFORGE :

- filtrage en **lumière linéaire** (un damier noir/blanc donne un gris 188 et non 128 : pas
  d'assombrissement au loin) ; addressage en boucle pour les textures répétables ;
- découpes (feuillage, grillages) : **couverture alpha conservée** à chaque niveau au seuil
  `alpha_ref` du test alpha du jeu ; dégradés d'alpha filtrés en alpha prémultiplié ;
- normal maps renormalisées (Z dans le bleu) ; DXT5nm et BC5 filtrées en linéaire, l'alpha d'une DXT5nm
  traité comme une donnée (ni prémultiplication, ni test de couverture) ; un original BC4/BC5 est réécrit en
  BGRA8 sans perte (jamais en DXT1) ; cartes de données filtrées sans conversion gamma.

Chaque niveau est ensuite compressé par **texconv** (DirectXTex, s'il est trouvé) ou par
l'encodeur intégré, le fichier est relu niveau par niveau et comparé à la chaîne de référence
(dimensions, PSNR, couverture alpha, luminosité). Format : celui de l'original par défaut
(`dds_format = "match"` : le jeu ou l'injecteur l'attend), sinon DXT1 sans alpha / DXT5 avec alpha.
**BC7 n'est pas lisible par Direct3D 9 ni 10** : il n'est utilisé qu'avec `target_api = "d3d11"`
et `allow_bc7 = true` (dgVoodoo2 + Special K), et seulement avec texconv.

Rapport : `qc_report.json` dans le dossier d'export (et `texforge_work/`), une entrée par texture
(décision, raisons, correctifs, avertissements, mesures avant/après, format, niveaux, PSNR).

```
== Contrôle qualité : 45 texture(s) — DDS Special K, encodeur texconv
[QC] ✔ Wall_04 validée (3/45) — SSIM 0.92, ΔE 1.6, raccord 1.02/1.12, détail ×5.27 → DXT1, 10 mip(s)
[QC] ✎ Rock_02 corrigée (4/45) — SSIM 0.95, ΔE 0.4 … → DXT1, 10 mip(s) | correctifs : verrou couleur ×1.00
[QC] ✘ Plank_03 rejetée (5/45) — SSIM 0.62, ΔE 3.1 … | structure 0.31 < 0.65; zone réinventée → version locale conservée
== QC terminé en 61 s — 38 validée(s), 4 corrigée(s), 3 rejetée(s), 0 locale(s) — DDS 412 Mo (originaux 31 Mo, ×13.3)
```

## 6. Piloter TEXFORGE depuis Node.js

`bridge/texforge-bridge.mjs` (Node 18+, sans dépendance) lance
`python -m texforge run --json-events` : la sortie standard ne contient qu'une ligne JSON par
événement, le tableau de bord lisible part sur la sortie d'erreur.

```js
import { TexforgeBridge } from './bridge/texforge-bridge.mjs';

const tf = new TexforgeBridge({ texforgeDir: 'C:/Outils/TEXFORGE', projectDir: 'C:/Remaster/TombRaider' });
tf.on('ui_done', (e) => console.log(`[UI] ${e.name} (${e.index}/${e.total})`));
tf.on('comfy_queued', (e) => console.log(`[COMFYUI] ${e.name} queued (${e.index}/${e.total})`));
tf.on('failed', (e) => console.error(`[ERREUR] ${e.name} : ${e.kind} ${e.message}`));
tf.on('qc_done', (e) => console.log(`[QC] ${e.name} ${e.label} ${e.format ?? ''}`));

await tf.scan();
const job = tf.run({ stages: ['ui', 'ai'] });   // job.stop() = arrêt propre
const summary = await job.done;                 // { code, ok, counts, failures, runEnd }

for (const kind of ['oom', 'timeout']) await tf.retry({ kind });
await tf.run({ stages: ['ai'] }).done;          // seconde passe

await tf.export({ out: 'C:/Jeux/TombRaider/SK_Res/inject/textures' }).done;   // QC + DDS
const report = await tf.report('C:/Jeux/TombRaider/SK_Res/inject/textures');
const rejected = report.textures.filter((t) => t.decision === 'rejected');
```

Exemple complet (tableau de bord, Ctrl+C, seconde passe automatique) :
`node bridge/example.mjs C:\Outils\TEXFORGE C:\Remaster\TombRaider`

| Événement | Champs utiles |
|---|---|
| `run_start` | `total`, `ui`, `local`, `ai`, `provider` |
| `stage_start` / `stage_end` | `stage`, `count` / `summary` |
| `ui_done` / `local_done` | `name`, `engine`, `k`, `index`, `total` |
| `comfy_queued` | `name`, `material`, `tiling`, `workflow`, `index`, `total` |
| `comfy_progress` | `message` (nœud en cours, étape du sampler ; non écrit dans events.jsonl) |
| `comfy_done` | `name`, `structure`, `attempt`, `seconds` |
| `retry` | `name`, `kind`, `attempt`, `message` |
| `fallback` | `name`, `engine`, `reason` |
| `failed` | `name`, `kind`, `message`, `stage` (texture abandonnée, sans repli) |
| `gpu_free` / `gpu_status` | `reason`, `vram_free_gb` |
| `server_down` / `server_back` | `message` |
| `stopped`, `run_end` | `summary`, `ui`, `local`, `ai`, `retried`, `seconds` |
| `qc_start` | `total`, `target`, `encoder` |
| `qc_done` | `name`, `decision` (validated, corrected, rejected, local, measured, error), `label`, `reasons`, `fixes`, `warnings`, `format`, `levels`, `psnr`, `index`, `total` |
| `qc_end` | `summary`, `validated`, `corrected`, `rejected`, `dds_mb`, `report` |

Chaque événement porte aussi `ts` (horodatage) et `texture` (identifiant stable de la texture).
Codes de sortie : 0 = terminé, 1 = terminé avec des échecs, 2 = arrêt fatal (workflow refusé,
serveur absent avec `on_server_down = "abort"`, configuration invalide) : `job.done` est alors rejetée.

Depuis Python, sans passer par la ligne de commande :

```python
from pathlib import Path
from texforge.config import load_config
from texforge.events import EventBus
from texforge.orchestrator import Orchestrator

cfg = load_config(Path("C:/Remaster/TombRaider/texforge.toml"))   # après un « texforge scan »
bus = EventBus(cfg["_work"], console=True)
summary = Orchestrator(cfg, bus).run(stages=["ui", "ai"])
```

## 7. Dépannage

| Symptôme | Cause probable | Correctif |
|---|---|---|
| Aucune texture analysée, seulement un `qc_report.json` en sortie (versions précédentes) | Entrée = dossier d'installation d'un jeu dont les textures sont dans des archives | Dump Special K d'abord, puis entrée = `SK_Res\dump\textures` ; supprimer le `qc_report.json` égaré |
| `Node 'TEXFORGE_TILING' not found` | Pack `ComfyUI-seamless-tiling` absent (le message cite le **titre** du nœud, son type est `SeamlessTile`) et TEXFORGE antérieur à 0.3.1 | Vérifier la ligne `[INFO] TEXFORGE 0.3.x lancé depuis …` (0.3.1 ou plus) au lancement : si elle manque ou indique un autre dossier, la copie lancée n'est pas la copie à jour. Ou installer le pack avec ComfyUI Manager |
| Visage ou peau rendu identique à l'original, « 0 ComfyUI » | Voulu : peau et visages sont agrandis sans IA générative (`skin_route = "local"`), l'IA redessinerait les traits | Régler Real-ESRGAN : les textures déjà agrandies par Lanczos faute de Real-ESRGAN sont refaites au lancement suivant (ligne `[LOCAL] … refaite(s) avec Real-ESRGAN`) |
| `FileNotFoundError` sur une texture | Fichier supprimé ou renommé depuis un scan précédent | Mis de côté automatiquement au scan suivant (statut `missing`) |
| `[COMFYUI] serveur injoignable` | ComfyUI non lancé ou autre port | Lancer ComfyUI, vérifier `server` |
| Arrêt « workflow refusé » | Nœud ou modèle manquant dans ComfyUI | Le message nomme le nœud ; installer le nœud ou corriger le nom du modèle |
| Beaucoup de `[RETRY] … oom` | VRAM saturée | `max_in_flight = 1`, `vram_threshold_gb = 3`, ou un checkpoint SD 1.5 |
| Icônes passées en ComfyUI | Score UI trop bas | `ui_threshold = 0.4` ou règle `mode = "ui"` |
| Pierres passées en UI | Texture très plate | `ui_threshold = 0.6` ou règle `mode = "ai"` |
| UI floue | Real-ESRGAN non trouvé | `realesrgan_exe`, puis `doctor` |
| Texture non remplacée en jeu | Format différent de l'original | `dds_format = "bgra8"` puis réexporter |
| Couleurs plus sombres/claires | Espace sRGB | `dds_format = "match"` |
| Plantage mémoire du jeu (32 bits) | Textures trop grandes | `max_side = 1024` ou `scale = 2` |
| Beaucoup de replis | Seuils trop stricts ou denoise trop fort | Ajuster `[quality]` ou `[materials.*]` |
| Beaucoup de « corrigée » (couleur) | Le modèle dérive la teinte | Baisser le denoise du matériau, ou `color_lock_always = 0.5` |
| QC « motifs décalés » au raccord | Raccord circulaire absent du workflow | Vérifier les nœuds `TEXFORGE_TILING` / `TEXFORGE_TILING_VAE` |
| Avertissement PSNR de compression | Dégradé ou texte en DXT | Règle `dds_format = "bgra8"` pour ces textures |
| Feuillage qui s'épaissit ou fond au loin | Seuil du test alpha du jeu différent | `alpha_ref = 0.25` à `0.75` selon le jeu |

## 8. Limites connues

- Un jeu en ligne qui vérifie ses fichiers ou bloque les injecteurs : à exclure.
- La génération n'est pas déterministe au pixel près : la validation et la revue existent pour cette raison.
- Le fournisseur cloud Higgsfield reste disponible (`provider.name = "higgsfield"`,
  `pip install higgsfield-client`, variable `HF_KEY`) mais n'est plus nécessaire.

## Structure

```
app_gui.py                   interface graphique
gui_compare.py               panneau avant / après de l'interface (vues synchronisées)
texforge/
  routing.py                 routage UI / ComfyUI / local, détection du matériau
  ui_upscale.py              pipeline UI : Real-ESRGAN / Waifu2x ncnn par lots, Lanczos
  orchestrator.py            étapes UI, local, ComfyUI : lots, VRAM, reprises, replis, arrêt propre
  comfyui_headless_client.py client ComfyUI asynchrone (HTTP + WebSocket)
  providers.py               ComfyUI (local), mock, Higgsfield (optionnel)
  events.py                  tableau de bord console + events.jsonl + JSON pour Node
  state.py                   base SQLite : textures, résultats, échecs
  remaster.py                chaîne complète (scan → génération → QC → DDS), réglages, progression globale
  eta.py                     estimation du temps : passages ComfyUI par texture, durées apprises (timings.json)
  updater.py                 mise à jour depuis GitHub : vérification, contrôles, sauvegarde, retour arrière
  release.json               liste des fichiers de la version et de leurs empreintes (envoi partiel refusé)
  prompt_enhance.py          prompt détaillé en anglais (Qwen3-4B dans ComfyUI, sinon LM Studio / Ollama)
  qc.py                      contrôle qualité final : mesures, verrou couleur, fondu du raccord, décision
  metrics.py                 CIELAB/ΔE2000, SSIM multi-échelle, palette, détail, raccords
  ddsbuild.py                format DDS, mipmaps (linéaire, couverture alpha), texconv ou intégré, relecture
  postprocess.py             QC + DDS par lots, rapport qc_report.json, mode autonome (postprocess)
  dds.py                     lecture/écriture DDS (en-têtes legacy et DX10), découpe par niveau
  imaging.py                 analyse, préparation, retour à la géométrie d'origine
  validate.py                contrôles automatiques
  pipeline.py                scan, règles, export
  review.py                  galerie HTML
  compare.py                 données du panneau avant / après (textures traitées, image et légende)
  unreal.py                  jeux Unreal : moteur, chiffrement et compression lus dans les en-têtes, conseils
  ba2.py                     archives BA2 (Fallout 4, Fallout 76, Starfield) : lecture, extraction des textures
  detect.py                  dossier de départ -> dossier des textures (Special K, SKIF, RTX Remix, BA2, fichiers)
  esrgan.py                  mode sans ComfyUI : Real-ESRGAN, garde anti-invention, verrou couleur, grain
  i18n.py                    langue active, tr(), contrôle des catalogues
  locales/en.py              traduction anglaise (ajouter une langue : copier ce fichier, traduire, LANGUAGES)
  selftest.py                test de bout en bout (dont catalogues de traduction complets)
app_gui.py                   interface graphique PySide6 (texforge_gui.bat)
workflows/                   workflows ComfyUI au format API (SDXL Tile, SDXL Tile+Canny, SD 1.5 Tile)
bridge/                      pont Node.js (texforge-bridge.mjs) et exemple
```
