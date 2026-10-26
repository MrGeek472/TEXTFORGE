"""English catalogue: French source text -> English.

Keys must match the French text passed to tr() / N_() exactly; {fields} and %-placeholders must be
kept. `python -m texforge selftest` lists missing or unused entries.
"""

MESSAGES: dict[str, str] = {
    # ---------------------------------------------------------------- interface (app_gui.py)
    "lancement impossible : {exc}": "could not start: {exc}",
    "génération en cours interrompue": "current generation interrupted",
    "file d'attente vidée": "queue cleared",
    "Parcourir…": "Browse…",
    "Rapport QC — {name}": "QC report — {name}",
    "textures": "textures",
    "validées": "validated",
    "corrigées": "corrected",
    "rejetées": "rejected",
    "locales": "local",
    "erreurs": "errors",
    "Mo": "MB",
    "Avec avertissement": "With warnings",
    "Corrigées": "Corrected",
    "Rejetées et erreurs": "Rejected and errors",
    "Toutes": "All",
    "Ouvrir le JSON": "Open JSON",
    "Décision": "Decision",
    "Raisons / correctifs / avertissements": "Reasons / fixes / warnings",
    "Texture": "Texture",
    "Mips": "Mips",
    "SDXL — ControlNet Tile (+ Canny pour les joints)": "SDXL — ControlNet Tile (+ Canny for joints)",
    "SD 1.5 — ControlNet Tile (moins de VRAM)": "SD 1.5 — ControlNet Tile (less VRAM)",
    "Workflow personnalisé (export API)": "Custom workflow (API export)",
    "Identique à l'original (recommandé)": "Same as original (recommended)",
    "Auto : BC1/DXT1 opaque · BC3/DXT5 alpha": "Auto: BC1/DXT1 opaque · BC3/DXT5 alpha",
    "BC1 / DXT1 (opaque, alpha 1 bit → DXT5)": "BC1 / DXT1 (opaque, 1-bit alpha → DXT5)",
    "BC3 / DXT5 (alpha)": "BC3 / DXT5 (alpha)",
    "BC7 haute qualité — Direct3D 11 uniquement": "BC7 high quality — Direct3D 11 only",
    "BGRA8 sans compression": "BGRA8 uncompressed",
    "DDS — injection Special K": "DDS — Special K injection",
    "PNG — RTX Remix": "PNG — RTX Remix",
    "Auto (texconv s'il est trouvé)": "Auto (texconv if found)",
    "texconv (DirectXTex)": "texconv (DirectXTex)",
    "Intégré": "Built-in",
    "(aucune)": "(none)",
    "(celui du workflow)": "(workflow's own)",
    "1 · SCAN & ROUTAGE": "1 · SCAN & ROUTING",
    "2 · GÉNÉRATION": "2 · GENERATION",
    "3 · QC & DDS": "3 · QC & DDS",
    "remaster de textures": "texture remaster",
    "[INFO] Langue : {name}": "[INFO] Language: {name}",
    "[INFO] TEXFORGE {version} prêt — choisis le dossier des textures extraites, puis LANCER LE PIPELINE.":
        "[INFO] TEXFORGE {version} ready — choose the folder of extracted textures, then RUN PIPELINE.",
    "Remaster IA local des textures DirectX 9 et antérieures — ComfyUI · ControlNet Tile · QC · DDS":
        "Local AI remaster of DirectX 9 and older textures — ComfyUI · ControlNet Tile · QC · DDS",
    "ComfyUI : non testé": "ComfyUI: not tested",
    "Langue de l'interface": "Interface language",
    "Langue de l'interface et des messages du traitement": "Language of the interface and of processing messages",
    "DOSSIERS": "FOLDERS",
    "Textures extraites, ex. C:\\Jeux\\MonJeu\\SK_Res\\dump\\textures":
        "Extracted textures, e.g. C:\\Games\\MyGame\\SK_Res\\dump\\textures",
    "Dossier des textures extraites": "Folder of extracted textures",
    "Sortie DDS, ex. C:\\Jeux\\MonJeu\\SK_Res\\inject\\textures": "DDS output, e.g. C:\\Games\\MyGame\\SK_Res\\inject\\textures",
    "Dossier de sortie": "Output Folder",
    "Les DDS finaux y sont écrits sous le nom d'origine, avec qc_report.json.\nPour Special K : SK_Res\\inject\\textures du jeu.":
        "Final DDS files are written here under their original names, with qc_report.json.\n"
        "For Special K: the game's SK_Res\\inject\\textures.",
    "Dossier d'entrée": "Input Folder",
    "Ouvrir": "Open",
    "Copier la commande": "Copy command",
    "Commande équivalente en ligne de commande (après un premier lancement)":
        "Equivalent command line (after a first run)",
    "Projet": "Project",
    "MODÈLE / CHECKPOINT COMFYUI": "COMFYUI MODEL / CHECKPOINT",
    "ComfyUI local (headless)": "Local ComfyUI (headless)",
    "Simulation sans GPU (test de la chaîne)": "Simulation without GPU (pipeline test)",
    "Moteur": "Engine",
    "Tester": "Test",
    "Serveur": "Server",
    "Famille": "Family",
    "Workflow API (*.json)": "API workflow (*.json)",
    "Workflow ComfyUI (format API)": "ComfyUI workflow (API format)",
    "facultatif : variante pour briques et carrelage": "optional: variant for bricks and tiles",
    "Workflow « joints »": "\"Joints\" workflow",
    "Workflow": "Workflow",
    "Workflow joints": "Joints workflow",
    "Checkpoint chargé par le nœud TEXFORGE_CHECKPOINT. « Tester » remplit la liste\navec les modèles réellement présents dans ComfyUI.":
        "Checkpoint loaded by the TEXFORGE_CHECKPOINT node. \"Test\" fills the list\n"
        "with the models actually present in ComfyUI.",
    "Relire la liste des modèles sur le serveur": "Reload the model list from the server",
    "LoRA insérée après le checkpoint (photoréalisme, matière…). Le nœud est ajouté\nautomatiquement au workflow ; le raccord circulaire reste actif.":
        "LoRA inserted after the checkpoint (photorealism, material…). The node is added\n"
        "to the workflow automatically; circular tiling stays active.",
    "Force LoRA": "LoRA strength",
    "RENDU && PROMPT": "RENDERING && PROMPT",
    "INTENSITÉ DE L'IA": "AI STRENGTH",
    "TAILLE DE SORTIE": "OUTPUT SIZE",
    "Force du ControlNet Tile : attache le rendu aux formes et aux couleurs de l'original.":
        "ControlNet Tile strength: ties the render to the shapes and colors of the original.",
    "Ajuster par matériau": "Adjust per material",
    "Coché : les curseurs décalent les préréglages de chaque matériau.\nDécoché : la même valeur pour toutes les textures.":
        "Checked: the sliders offset each material's presets.\nUnchecked: the same value for every texture.",
    "max {v} px": "max {v} px",
    "Côté maximal. Jeux 32 bits : 2048 ou moins (mémoire du processus du jeu).":
        "Maximum side. 32-bit games: 2048 or less (memory of the game process).",
    "Agrandissement": "Upscale",
    "toutes": "all",
    "Lot d'essai": "Test batch",
    "DDS FORMAT && COMPRESSION": "DDS FORMAT && COMPRESSION",
    "Cible": "Target",
    "« Identique à l'original » : le jeu (ou l'injecteur) attend ce format.\nBC7 n'est lisible qu'en Direct3D 11 (dgVoodoo2 + Special K).":
        "\"Same as original\": the game (or the injector) expects this format.\n"
        "BC7 can only be read by Direct3D 11 (dgVoodoo2 + Special K).",
    "Format": "Format",
    "BC7 n'existe pas en DirectX 9/10 : à n'utiliser qu'avec dgVoodoo2 + Special K (Direct3D 11), et texconv.":
        "BC7 does not exist in DirectX 9/10: use it only with dgVoodoo2 + Special K (Direct3D 11), and texconv.",
    "Encodeur": "Encoder",
    "texconv.exe (sinon tools\\texconv.exe ou PATH)": "texconv.exe (otherwise tools\\texconv.exe or PATH)",
    "OPTIONS && ROUTAGE": "OPTIONS && ROUTING",
    "Routage automatique HUD / UI": "Automatic HUD / UI routing",
    "Polices, icônes, HUD et cartes partent vers Real-ESRGAN / Waifu2x ; les matières\nvers ComfyUI. Décoché : tout ce qui n'est pas une donnée part vers ComfyUI.":
        "Fonts, icons, HUD and maps go to Real-ESRGAN / Waifu2x; materials\n"
        "go to ComfyUI. Unchecked: everything that is not data goes to ComfyUI.",
    "realesrgan-ncnn-vulkan.exe (sinon Lanczos)": "realesrgan-ncnn-vulkan.exe (otherwise Lanczos)",
    "Programme (*.exe)": "Program (*.exe)",
    "Seamless tiling": "Seamless tiling",
    "Les textures détectées comme répétables sont générées avec un raccord circulaire,\npuis vérifiées au QC. Les icônes, atlas et décals ne sont pas concernés.":
        "Textures detected as tileable are generated with circular padding,\n"
        "then checked by QC. Icons, atlases and decals are not affected.",
    "Forcer sur toutes les textures (déconseillé)": "Force on every texture (not recommended)",
    "Le raccord circulaire fait se toucher les bords opposés : sur un atlas,\nle contenu d'un bord déborde sur l'autre. À réserver à un dossier de matières.":
        "Circular padding makes opposite edges touch: on an atlas,\n"
        "one edge's content bleeds into the other. Keep it for a folder of materials.",
    "Génération automatique des mipmaps": "Automatic mipmap generation",
    "Décoché : mipmaps seulement si l'original en avait.": "Unchecked: mipmaps only if the original had them.",
    "Contrôle qualité SSIM / préservation de la DA": "SSIM quality control / art direction preservation",
    "Structure, couleurs (ΔE2000), palette, raccord et transparence comparés à l'original\nramené à sa résolution. Échec : correctif automatique ou version locale conservée.":
        "Structure, colors (ΔE2000), palette, seams and transparency compared with the original\n"
        "at its own resolution. Failure: automatic fix, or the local version is kept.",
    "Correctifs automatiques": "Automatic fixes",
    "Verrou couleur basses fréquences et fondu des raccords avant de rejeter une texture.":
        "Low-frequency color lock and seam blending before a texture is rejected.",
    "Régénérer si les réglages ont changé": "Regenerate if settings changed",
    "Après un changement de modèle, de denoise ou de ControlNet, les textures déjà\ngénérées sont refaites. Les choix faits à la main dans la galerie sont conservés.":
        "After a change of model, denoise or ControlNet, textures already\n"
        "generated are redone. Choices made by hand in the gallery are kept.",
    "Upscale local rapide (Real-ESRGAN / Waifu2x), sans IA générative : les textes restent lisibles.":
        "Fast local upscale (Real-ESRGAN / Waifu2x), no generative AI: text stays readable.",
    "Raccord parfait des bords sur les textures répétables détectées (sols, murs, terrains).":
        "Perfectly matching edges on detected tileable textures (floors, walls, terrain).",
    "Chaîne complète filtrée en lumière linéaire, couverture alpha conservée (feuillage, grilles).":
        "Full chain filtered in linear light, alpha coverage preserved (foliage, grates).",
    "Structure, couleurs, palette, raccord et transparence comparés à l'original.":
        "Structure, colors, palette, seams and transparency compared with the original.",
    "▶  LANCER LE PIPELINE": "▶  RUN PIPELINE",
    "En attente": "Waiting",
    "REPLIS": "FALLBACKS",
    "ERREURS": "ERRORS",
    "CONSOLE": "CONSOLE",
    "Alertes et étapes seulement": "Alerts and stages only",
    "Tout afficher": "Show everything",
    "Effacer": "Clear",
    "Enregistrer…": "Save…",
    "📂  Ouvrir la sortie": "📂  Open output",
    "📋  Rapport QC": "📋  QC report",
    "🖼  Galerie avant / après": "🖼  Before / after gallery",
    "Encodeur intégré : DXT1, DXT3, DXT5 (pas de BC7).": "Built-in encoder: DXT1, DXT3, DXT5 (no BC7).",
    "✓ texconv trouvé : {exe}": "✓ texconv found: {exe}",
    "texconv absent : l'encodeur intégré prendra le relais (DXT1/3/5).":
        "texconv missing: the built-in encoder takes over (DXT1/3/5).",
    "bois": "wood",
    "maçonnerie": "masonry",
    "métal": "metal",
    "pierre": "stone",
    "peau": "skin",
    "tissu": "fabric",
    "végétation": "vegetation",
    "denoise / ControlNet effectifs : {txt}": "effective denoise / ControlNet: {txt}",
    "Simulation : ComfyUI non utilisé": "Simulation: ComfyUI not used",
    "ComfyUI : test…": "ComfyUI: testing…",
    "● ComfyUI injoignable": "● ComfyUI unreachable",
    ", {gb:.1f} Go libres": ", {gb:.1f} GB free",
    "le dossier de sortie est à l'intérieur du dossier d'entrée : les DDS produits seraient rescannés comme des textures du jeu":
        "the output folder is inside the input folder: the DDS files produced would be rescanned as game textures",
    "Impossible de lancer :": "Cannot start:",
    "BC7 n'est lisible qu'en Direct3D 11.\n\nLe jeu doit passer par dgVoodoo2 + Special K, et texconv doit être installé (sinon BC3). Continuer ?":
        "BC7 can only be read by Direct3D 11.\n\nThe game must go through dgVoodoo2 + Special K, and texconv must be "
        "installed (otherwise BC3). Continue?",
    "ComfyUI ne répond pas à {server}.": "ComfyUI is not responding at {server}.",
    "Annuler": "Cancel",
    "Démarrage…": "Starting…",
    "══ Lancement — {src} → {dst}": "══ Run — {src} → {dst}",
    "■  FORCER L'ARRÊT": "■  FORCE STOP",
    "■ Arrêt demandé : les textures en cours se terminent, l'état est enregistré (relancer reprend où l'on s'est arrêté).":
        "■ Stop requested: textures in progress finish and the state is saved (running again resumes where it stopped).",
    "Arrêter immédiatement le traitement ?\n\nLes textures en cours sont perdues (elles seront refaites au prochain lancement) et la file ComfyUI est vidée.":
        "Stop processing immediately?\n\nTextures in progress are lost (they will be redone on the next run) and the "
        "ComfyUI queue is cleared.",
    "■ Arrêt forcé du traitement.": "■ Processing force-stopped.",
    "■  ANNULER": "■  CANCEL",
    "ComfyUI : {msg}": "ComfyUI: {msg}",
    "{label} : {done}/{total}": "{label}: {done}/{total}",
    " — reste ~{t} pour cette étape": " — ~{t} left for this stage",
    "Pipeline terminé": "Pipeline finished",
    "Terminé avec des échecs (voir la console)": "Finished with failures (see the console)",
    "Arrêté — relancer reprend où l'on s'est arrêté": "Stopped — running again resumes where it stopped",
    "Interrompu par une erreur (voir la console)": "Interrupted by an error (see the console)",
    "Processus terminé avec le code {code}": "Process ended with code {code}",
    "100 %  —  terminé": "100 %  —  done",
    "{p:.0f} %  —  arrêté": "{p:.0f} %  —  stopped",
    "[ERREUR] le processus s'est terminé avec le code {code}": "[ERROR] the process ended with code {code}",
    "[INFO] Rapport QC : {path}": "[INFO] QC report: {path}",
    "Aucun rapport QC pour ce dossier de sortie : lance d'abord le pipeline.":
        "No QC report for this output folder: run the pipeline first.",
    "Aucun projet pour ce dossier d'entrée : lance d'abord le pipeline.":
        "No project for this input folder: run the pipeline first.",
    "[INFO] Génération de la galerie avant / après…": "[INFO] Building the before / after gallery…",
    "[ERREUR] galerie : {err}": "[ERROR] gallery: {err}",
    "[INFO] Galerie : {page}": "[INFO] Gallery: {page}",
    "Copié : {cmd}": "Copied: {cmd}",
    "Enregistrer le journal": "Save the log",
    "Texte (*.txt)": "Text (*.txt)",
    "texforge_journal.txt": "texforge_log.txt",
    "La langue se change une fois le traitement terminé.": "The language can be changed once processing has finished.",
    "Un traitement est en cours. L'arrêter et quitter ?\n\nL'état est enregistré : relancer reprendra où l'on s'est arrêté.":
        "Processing is running. Stop it and quit?\n\nThe state is saved: running again will resume where it stopped.",

    # ---------------------------------------------------------------- détection du dossier d'entrée
    "🔍 Détecter": "🔍 Detect",
    "🔍 Détecter ({n})": "🔍 Detect ({n})",
    "🔍 Recherche…": "🔍 Searching…",
    "Recherche en cours…": "Searching…",
    "Rechercher à nouveau": "Search again",
    "Dossiers de textures extraites trouvés sur ce PC (Special K, RTX Remix),\ndu plus récent au plus ancien":
        "Extracted texture folders found on this PC (Special K, RTX Remix),\nmost recent first",
    "{game} — {count} texture(s) · {when} · {kind}": "{game} — {count} texture(s) · {when} · {kind}",
    "Aucun dump trouvé : active le dump des textures dans Special K": "No dump found: enable texture dumping in Special K",
    "[INFO] Détection du dossier de textures impossible : {err}": "[INFO] Could not detect the texture folder: {err}",
    "[INFO] Aucun dump de textures trouvé. Dans Special K (Ctrl+Maj+Retour arrière), active le dump des textures, joue, puis « 🔍 Détecter » → Rechercher à nouveau. Le dossier d'installation seul ne marche que si le jeu y range ses textures en fichiers séparés.":
        "[INFO] No texture dump found. In Special K (Ctrl+Shift+Backspace), enable texture dumping, play, then "
        "\"🔍 Detect\" → Search again. The install folder alone only works if the game keeps its textures as separate files.",
    "[INFO] {n} autre(s) jeu(x) avec des textures extraites : bouton « 🔍 Détecter ».":
        "[INFO] {n} other game(s) with extracted textures: \"🔍 Detect\" button.",
    "[INFO] Dossier de textures détecté : {game} — {count} texture(s) ({path})":
        "[INFO] Texture folder detected: {game} — {count} texture(s) ({path})",
    "[INFO] Dossier de textures : {game} — {count} texture(s) ({path})":
        "[INFO] Texture folder: {game} — {count} texture(s) ({path})",
    "[INFO] Cible : {target}": "[INFO] Target: {target}",
    "[INFO] Sortie : dossier d'injection Special K du jeu ({path})":
        "[INFO] Output: the game's Special K injection folder ({path})",
    "Aucun dossier de textures extraites trouvé ({seconds:.1f} s).": "No extracted texture folder found ({seconds:.1f} s).",
    "Special K : active le dump des textures en jeu ; sinon indique le dossier à la main ou ajoute ta bibliothèque de jeux dans TEXFORGE_SEARCH.":
        "Special K: enable texture dumping in game; otherwise give the folder by hand or add your game library to "
        "TEXFORGE_SEARCH.",
    "{n} dossier(s) trouvé(s) en {seconds:.1f} s, du plus récent au plus ancien :":
        "{n} folder(s) found in {seconds:.1f} s, most recent first:",
    "{count} texture(s), dernière le {when}": "{count} texture(s), latest on {when}",
    "injection : {path}": "injection: {path}",
    "Steam": "Steam",
    "Epic": "Epic",
    "GOG": "GOG",
    "programme installé": "installed program",
    "dossier de jeux": "games folder",
    "profil Special K": "Special K profile",
    "déjà utilisé": "used before",
    "dossier choisi": "chosen folder",

    # ---------------------------------------------------------------- mode sans ComfyUI (Real-ESRGAN)
    "Real-ESRGAN — sans ComfyUI (rendu réaliste)": "Real-ESRGAN — without ComfyUI (realistic render)",
    "ComfyUI : remaster génératif (SDXL + ControlNet Tile), la matière est réinventée.\nReal-ESRGAN : sans ComfyUI ; agrandissement net, couleurs de l'original,\ngrain de matière dosé d'après la texture, garde contre les motifs inventés.\nChoisi automatiquement si ComfyUI ne répond pas (tant que tu n'imposes pas de moteur).":
        "ComfyUI: generative remaster (SDXL + ControlNet Tile), the material is reinvented.\n"
        "Real-ESRGAN: no ComfyUI needed; sharp upscale, original colors,\n"
        "material grain dosed from the texture, guard against invented patterns.\n"
        "Picked automatically when ComfyUI does not answer (unless you force an engine).",
    "Installer Real-ESRGAN": "Install Real-ESRGAN",
    "Télécharge realesrgan-ncnn-vulkan (version officielle de GitHub, xinntao/Real-ESRGAN,\nempreinte SHA-256 vérifiée) dans le dossier tools de TEXFORGE.":
        "Downloads realesrgan-ncnn-vulkan (official GitHub release, xinntao/Real-ESRGAN,\n"
        "SHA-256 checked) into TEXFORGE's tools folder.",
    "Mode Real-ESRGAN : grain de matière ajouté là où le modèle a lissé.\n0 = aucun ; 1 = dose mesurée d'après la texture ; 1,5 = appuyé.":
        "Real-ESRGAN mode: material grain added where the model smoothed things out.\n"
        "0 = none; 1 = dose measured from the texture; 1.5 = strong.",
    "Micro-détail": "Micro-detail",
    " textures IA": " AI textures",
    "Pour un premier essai : limite le nombre de matières traitées par ComfyUI ou Real-ESRGAN (0 = toutes).":
        "For a first try: limits the number of materials processed by ComfyUI or Real-ESRGAN (0 = all).",
    "[INFO] Téléchargement de Real-ESRGAN ({name}, {mb:.0f} Mo) depuis GitHub…":
        "[INFO] Downloading Real-ESRGAN ({name}, {mb:.0f} MB) from GitHub…",
    "✓ installé : {path}": "✓ installed: {path}",
    "⚠ absent : nécessaire pour ce moteur": "⚠ missing: required by this engine",
    "absent (sert aussi aux HUD et aux replis)": "missing (also used for HUD and fallbacks)",
    "téléchargement : {pct} %": "downloading: {pct} %",
    "[ERREUR] installation de Real-ESRGAN : {err}": "[ERROR] Real-ESRGAN installation: {err}",
    "Installation de Real-ESRGAN impossible :\n\n{err}": "Could not install Real-ESRGAN:\n\n{err}",
    "[INFO] Real-ESRGAN installé : {path}": "[INFO] Real-ESRGAN installed: {path}",
    "[INFO] ComfyUI ne répond pas : moteur Real-ESRGAN (rendu réaliste, sans IA générative). Lance ComfyUI puis « Tester » pour revenir au remaster génératif.":
        "[INFO] ComfyUI is not answering: Real-ESRGAN engine (realistic render, no generative AI). Start ComfyUI, "
        "then \"Test\" to go back to the generative remaster.",
    "[INFO] ComfyUI répond : moteur ComfyUI (remaster génératif).": "[INFO] ComfyUI answers: ComfyUI engine (generative remaster).",
    "● Real-ESRGAN (sans ComfyUI)": "● Real-ESRGAN (no ComfyUI)",
    "Real-ESRGAN n'est pas installé.\n\nLe télécharger maintenant ? ({mb:.0f} Mo, version officielle de GitHub, xinntao/Real-ESRGAN, empreinte vérifiée)\n\n{name}":
        "Real-ESRGAN is not installed.\n\nDownload it now? ({mb:.0f} MB, official GitHub release, xinntao/Real-ESRGAN, "
        "checksum verified)\n\n{name}",
    "« Utiliser Real-ESRGAN » : rendu réaliste sans ComfyUI (agrandissement net, couleurs\nde l'original, grain de matière), sans IA générative.\n\n« Attendre ComfyUI » : lance-le (run_nvidia_gpu.bat ou python main.py --port 8188) ;\nTEXFORGE l'attend 2 minutes.":
        "\"Use Real-ESRGAN\": realistic render without ComfyUI (sharp upscale, original\ncolors, material grain), "
        "no generative AI.\n\n\"Wait for ComfyUI\": start it (run_nvidia_gpu.bat or python main.py --port 8188);\n"
        "TEXFORGE waits for it 2 minutes.",
    "Utiliser Real-ESRGAN": "Use Real-ESRGAN",
    "Attendre ComfyUI": "Wait for ComfyUI",
    "Téléchargement de {name} ({mb:.0f} Mo) depuis GitHub (xinntao/Real-ESRGAN)…":
        "Downloading {name} ({mb:.0f} MB) from GitHub (xinntao/Real-ESRGAN)…",
    "✓ Real-ESRGAN installé : {path}": "✓ Real-ESRGAN installed: {path}",
    "Real-ESRGAN déjà présent : {path}": "Real-ESRGAN already present: {path}",
    "archive Real-ESRGAN non conforme ({got} octets reçus, {size} attendus, ou empreinte SHA-256 différente de la version officielle) : rien n'a été installé":
        "Real-ESRGAN archive does not match ({got} bytes received, {size} expected, or SHA-256 differs from the "
        "official release): nothing was installed",
    "archive Real-ESRGAN sans exécutable {name}": "Real-ESRGAN archive without the {name} executable",
    "téléchargement de Real-ESRGAN impossible : {err}": "could not download Real-ESRGAN: {err}",
    "Real-ESRGAN : code {code} : {tail}": "Real-ESRGAN: code {code}: {tail}",
    "[ESRGAN] {model} ×{scale} : {n} image(s) en {seconds:.1f} s": "[ESRGAN] {model} ×{scale}: {n} image(s) in {seconds:.1f} s",
    "Real-ESRGAN introuvable : bouton « Installer Real-ESRGAN » ou python -m texforge install-realesrgan":
        "Real-ESRGAN not found: \"Install Real-ESRGAN\" button or python -m texforge install-realesrgan",
    "Real-ESRGAN : délai de {seconds:.0f} s dépassé": "Real-ESRGAN: {seconds:.0f} s timeout exceeded",
    "Real-ESRGAN : lancement impossible ({err})": "Real-ESRGAN: could not start ({err})",
    "Real-ESRGAN : aucune sortie": "Real-ESRGAN: no output",
    "[ESRGAN] {name} agrandi ×{k} — {material}, part prudente {guard:.0%}, grain {grain:.1f}{pos}":
        "[ESRGAN] {name} upscaled ×{k} — {material}, conservative share {guard:.0%}, grain {grain:.1f}{pos}",
    "{ui} UI, {local} local, {ai} {engine}, {fallback} replis, {retried} reprises, {errors} échecs":
        "{ui} UI, {local} local, {ai} {engine}, {fallback} fallbacks, {retried} retries, {errors} failures",
    "[ESRGAN] {exe}": "[ESRGAN] {exe}",
    "simulation": "simulation",
    "{n} texture(s) agrandie(s) par Real-ESRGAN": "{n} texture(s) upscaled by Real-ESRGAN",
    "⚠ ComfyUI injoignable : les {n} matières passent par Real-ESRGAN (rendu réaliste, sans IA générative). Avec ComfyUI : texforge retry --kind server, puis relancer.":
        "⚠ ComfyUI unreachable: the {n} materials go through Real-ESRGAN (realistic render, no generative AI). "
        "With ComfyUI: texforge retry --kind server, then run again.",
    "fournisseur inconnu : {name} (attendu : comfyui, esrgan, mock ou higgsfield)":
        "unknown provider: {name} (expected: comfyui, esrgan, mock or higgsfield)",
    "ComfyUI injoignable : matières agrandies sans IA générative": "ComfyUI unreachable: materials upscaled without generative AI",
    "moteur inconnu : {name}": "unknown engine: {name}",
    "{ui} UI, {local} local, {ai} {engine}, {fallback} replis, {errors} échecs":
        "{ui} UI, {local} local, {ai} {engine}, {fallback} fallbacks, {errors} failures",

    # ---------------------------------------------------------------- client ComfyUI / SD WebUI
    "réponse JSON invalide pour {what} : {body!r}": "invalid JSON response for {what}: {body!r}",
    "image illisible : {exc}": "unreadable image: {exc}",
    "image renvoyée illisible : {exc}": "returned image is unreadable: {exc}",
    "workflow introuvable : {path}": "workflow not found: {path}",
    "JSON invalide dans {path} (ligne {lineno}) : {msg}": "invalid JSON in {path} (line {lineno}): {msg}",
    "ce fichier est au format « interface » de ComfyUI : réexporte-le avec Workflow → Export (API)":
        "this file is in ComfyUI's \"UI\" format: export it again with Workflow → Export (API)",
    "workflow vide ou mal formé (objet JSON attendu)": "empty or malformed workflow (JSON object expected)",
    "nœud {nid!r} mal formé : « class_type » et « inputs » attendus (format API)":
        "malformed node {nid!r}: \"class_type\" and \"inputs\" expected (API format)",
    "nœud d'entrée {load_node_id!r} absent du workflow": "input node {load_node_id!r} missing from the workflow",
    "aucun nœud LoadImage dans le workflow": "no LoadImage node in the workflow",
    "plusieurs nœuds LoadImage ({nodes}) : renomme celui de la texture « {tag} » ou passe --load-node ID":
        "several LoadImage nodes ({nodes}): rename the texture one to \"{tag}\" or pass --load-node ID",
    "nœud(s) de sortie absent(s) : {nodes}": "missing output node(s): {nodes}",
    "aucun nœud SaveImage (ni PreviewImage) : le workflow ne produit rien":
        "no SaveImage (or PreviewImage) node: the workflow produces nothing",
    "pas de SaveImage : lecture depuis le dossier temporaire de ComfyUI (PreviewImage)":
        "no SaveImage: reading from ComfyUI's temporary folder (PreviewImage)",
    "--set : plusieurs nœuds portent le titre {ref!r} ({nodes})": "--set: several nodes have the title {ref!r} ({nodes})",
    "--set : aucun nœud d'id ou de titre {ref!r} dans le workflow": "--set: no node with id or title {ref!r} in the workflow",
    "moteur non démarré (appeler start())": "engine not started (call start())",
    "{label} injoignable ({base_url}) : {exc}": "{label} unreachable ({base_url}): {exc}",
    "{method} {path} : délai réseau dépassé": "{method} {path}: network timeout",
    "{method} {path} : réponse tronquée ({exc})": "{method} {path}: truncated response ({exc})",
    "/system_stats a répondu {status} : est-ce bien ComfyUI ?": "/system_stats answered {status}: is this really ComfyUI?",
    "WebSocket connecté": "WebSocket connected",
    "WebSocket indisponible (suivi par sondage)": "WebSocket unavailable (polling instead)",
    "ComfyUI {version} — {gpu}, VRAM libre {free}/{total} Go — {ws}": "ComfyUI {version} — {gpu}, free VRAM {free}/{total} GB — {ws}",
    "/free a répondu %s": "/free answered %s",
    "le workflow n'a produit aucune image": "the workflow produced no image",
    "envoi de l'image refusé ({status}) : {body!r}": "image upload refused ({status}): {body!r}",
    "réponse /upload/image sans nom de fichier": "/upload/image response without a file name",
    "sorties partiellement invalides : %s": "partially invalid outputs: %s",
    "réponse /prompt sans prompt_id": "/prompt response without prompt_id",
    "délai de {timeout:.0f} s dépassé (requête annulée)": "{timeout:.0f} s timeout exceeded (request cancelled)",
    "téléchargement de {filename} impossible ({status})": "could not download {filename} ({status})",
    "WebSocket fermé, reconnexion dans %.0f s": "WebSocket closed, reconnecting in %.0f s",
    "exécution interrompue": "execution interrupted",
    " · file {n}": " · queue {n}",
    "/prompt a répondu {status} : {body}": "/prompt answered {status}: {body}",
    "nœud {nid} ({cls}) : {message} {details}": "node {nid} ({cls}): {message} {details}",
    "raison inconnue": "unknown reason",
    "workflow refusé ({status}) : ": "workflow refused ({status}): ",
    "échec signalé dans l'historique (voir console ComfyUI)": "failure reported in the history (see the ComfyUI console)",
    " — nœud {node} ({kind})": " — node {node} ({kind})",
    "paramètres introuvables : {path}": "parameters not found: {path}",
    "paramètres img2img : objet JSON attendu": "img2img parameters: JSON object expected",
    "ce fichier est un workflow ComfyUI : utilise --backend comfyui": "this file is a ComfyUI workflow: use --backend comfyui",
    "API absente : relance la WebUI avec l'option --api": "API missing: restart the WebUI with the --api option",
    "/sdapi/v1/progress a répondu {status}": "/sdapi/v1/progress answered {status}",
    "SD WebUI — modèle {model}, VRAM libre {free} Go": "SD WebUI — model {model}, free VRAM {free} GB",
    "/sdapi/v1/unload-checkpoint a répondu %s": "/sdapi/v1/unload-checkpoint answered %s",
    "rechargement du modèle impossible ({status})": "could not reload the model ({status})",
    "délai de {timeout:.0f} s dépassé (génération interrompue)": "{timeout:.0f} s timeout exceeded (generation interrupted)",
    "la WebUI n'a renvoyé aucune image": "the WebUI returned no image",
    " · pas {step}/{steps}": " · step {step}/{steps}",
    " · reste ~{eta:.0f} s": " · ~{eta:.0f} s left",
    "paramètres refusés par la WebUI (422) : {msg}": "parameters refused by the WebUI (422): {msg}",
    "point d'accès introuvable (404) : lance la WebUI avec --api": "endpoint not found (404): start the WebUI with --api",
    "erreur WebUI ({status}) : {msg}": "WebUI error ({status}): {msg}",
    "moteur inconnu : {backend}": "unknown engine: {backend}",
    "dossier d'entrée introuvable : {in_root}": "input folder not found: {in_root}",
    "workflow invalide : {exc}": "invalid workflow: {exc}",

    # ---------------------------------------------------------------- configuration
    "configuration introuvable : {path} (lance d'abord : texforge init)": "configuration not found: {path} (run first: texforge init)",
    "erreur de syntaxe dans {path} : {exc}": "syntax error in {path}: {exc}",

    # ---------------------------------------------------------------- lecture / écriture DDS
    "fichier DDS invalide (signature absente)": "invalid DDS file (missing signature)",
    "taille d'en-tête inattendue ({size})": "unexpected header size ({size})",
    "en-tête DX10 tronqué": "truncated DX10 header",
    "profondeur non gérée ({bits} bits)": "unsupported bit depth ({bits} bits)",
    "cubemap ou texture volumique : non traitée": "cubemap or volume texture: not processed",
    "décodage impossible du format {format} : {exc}": "cannot decode format {format}: {exc}",
    "encodeur intégré : format {codec} non géré": "built-in encoder: format {codec} not supported",
    "format d'écriture non géré : {fmt}": "unsupported output format: {fmt}",
    "niveau {i} ({lw}x{lh}) : {n} octets au lieu de {expected}": "level {i} ({lw}x{lh}): {n} bytes instead of {expected}",
    "cubemap ou texture volumique : non gérée": "cubemap or volume texture: not supported",
    "format {format} : découpage non géré": "format {format}: level splitting not supported",
    "fichier tronqué au niveau {i}": "file truncated at level {i}",

    # ---------------------------------------------------------------- construction DDS
    "format d'origine {src_fmt} non inscriptible : {fmt}": "original format {src_fmt} cannot be written: {fmt}",
    "original non DDS : BGRA8 sans perte (dds_format = \"auto\" pour compresser en DXT)":
        "non-DDS original: lossless BGRA8 (dds_format = \"auto\" to compress to DXT)",
    "dds_format inconnu : {policy}": "unknown dds_format: {policy}",
    "alpha en dégradé : DXT1 → DXT5": "gradient alpha: DXT1 → DXT5",
    "BC7 refusé (Direct3D 9/10 ne le lit pas ; target_api = \"d3d11\" et allow_bc7 = true pour dgVoodoo2 + Special K)":
        "BC7 refused (Direct3D 9/10 cannot read it; target_api = \"d3d11\" and allow_bc7 = true for dgVoodoo2 + Special K)",
    "BC7 impossible sans texconv : BC3": "BC7 impossible without texconv: BC3",
    "{fmt} demande un en-tête DX10 : {legacy} (D3D9)": "{fmt} needs a DX10 header: {legacy} (D3D9)",
    "{fmt} sans alpha : transparence perdue": "{fmt} has no alpha: transparency lost",
    "{w}x{h} → {W}x{H} (multiple de 4 exigé par DXT)": "{w}x{h} → {W}x{H} (DXT requires multiples of 4)",
    "encodeur intégré : {codec} non géré": "built-in encoder: {codec} not supported",
    "texconv : délai de {timeout:.0f} s dépassé": "texconv: {timeout:.0f} s timeout exceeded",
    "texconv : lancement impossible ({exc})": "texconv: could not start ({exc})",
    "texconv code {code} : {tail}": "texconv code {code}: {tail}",
    "[DDS] texconv {fmt} : {n} niveau(x) en {seconds:.1f} s": "[DDS] texconv {fmt}: {n} level(s) in {seconds:.1f} s",
    "texconv : sortie absente pour {stem}": "texconv: missing output for {stem}",
    "texconv : format {got} reçu au lieu de {codec}": "texconv: got format {got} instead of {codec}",
    "texconv introuvable : renseigne [qc] texconv_exe ou place texconv.exe dans tools/":
        "texconv not found: set [qc] texconv_exe or put texconv.exe in tools/",
    "relecture : {exc}": "read-back: {exc}",
    "en-tête {width}x{height}/{n} niv. au lieu de {width2}x{height2}/{n2}":
        "header {width}x{height}/{n} levels instead of {width2}x{height2}/{n2}",
    "format relu {codec} au lieu de {codec2}": "read-back format {codec} instead of {codec2}",
    "niveau {i} illisible : {exc}": "level {i} unreadable: {exc}",
    "niveau {i} : taille {shape1}x{shape0}": "level {i}: size {shape1}x{shape0}",
    "[DDS] {exc} — repli sur l'encodeur intégré": "[DDS] {exc} — falling back to the built-in encoder",
    "texconv indisponible : BC3 au lieu de BC7": "texconv unavailable: BC3 instead of BC7",
    "aucun encodeur disponible pour {codec}": "no encoder available for {codec}",

    # ---------------------------------------------------------------- console (events.py)
    "== TEXFORGE : {total} texture(s) à traiter — moteur IA : {provider}":
        "== TEXFORGE: {total} texture(s) to process — AI engine: {provider}",
    "── Étape {stage} : {count} texture(s)": "── Stage {stage}: {count} texture(s)",
    "── Fin {stage} : {summary}": "── End of {stage}: {summary}",
    "[UI] {name} traité localement ({engine}, ×{k}){pos}": "[UI] {name} processed locally ({engine}, ×{k}){pos}",
    "[LOCAL] {name} agrandi ({engine}, ×{k}){pos}": "[LOCAL] {name} upscaled ({engine}, ×{k}){pos}",
    "[COMFYUI] {name} en file{pos} — {material}, tiling {tiling}": "[COMFYUI] {name} queued{pos} — {material}, tiling {tiling}",
    "[COMFYUI] {name} validé{pos} — structure {structure}, essai {attempt}, {seconds:.1f} s":
        "[COMFYUI] {name} accepted{pos} — structure {structure}, attempt {attempt}, {seconds:.1f} s",
    "[REPLI] {name} → {engine} ({reason}){pos}": "[FALLBACK] {name} → {engine} ({reason}){pos}",
    "[RETRY] {name} : {message} — nouvel essai {attempt}": "[RETRY] {name}: {message} — new attempt {attempt}",
    "[ERREUR] {name} : {message} [{kind}]": "[ERROR] {name}: {message} [{kind}]",
    "[GPU] VRAM libérée ({reason})": "[GPU] VRAM freed ({reason})",
    " — libre avant : {gb:.1f} Go": " — free before: {gb:.1f} GB",
    "[COMFYUI] serveur injoignable — {message}": "[COMFYUI] server unreachable — {message}",
    "[COMFYUI] serveur de nouveau disponible": "[COMFYUI] server available again",
    "■ Arrêt : {message}": "■ Stop: {message}",
    "== Terminé en {seconds:.0f} s — {summary}": "== Finished in {seconds:.0f} s — {summary}",
    ", LoRA {lora}": ", LoRA {lora}",
    "[INFO] Chaîne complète — entrée {input}": "[INFO] Full pipeline — input {input}",
    "[INFO] sortie {output} — moteur {provider}, modèle {model}, checkpoint {checkpoint}":
        "[INFO] output {output} — engine {provider}, model {model}, checkpoint {checkpoint}",
    "[INFO] projet : {work}": "[INFO] project: {work}",
    "══ ÉTAPE {index}/{total} : {label}": "══ STAGE {index}/{total}: {label}",
    "⚠ Pipeline terminé avec des échecs": "⚠ Pipeline finished with failures",
    "✔ Pipeline terminé": "✔ Pipeline finished",
    "■ Pipeline arrêté": "■ Pipeline stopped",
    "✗ Pipeline interrompu": "✗ Pipeline interrupted",
    "Fin": "End",
    "══ {head} en {seconds:.0f} s": "══ {head} in {seconds:.0f} s",
    "DDS Special K": "Special K DDS",
    "PNG RTX Remix": "RTX Remix PNG",
    "sans export": "no export",
    "== Contrôle qualité : {total} texture(s) — {target}": "== Quality control: {total} texture(s) — {target}",
    ", encodeur {encoder}": ", encoder {encoder}",
    " (QC désactivé)": " (QC disabled)",
    " → {format}, {levels} mip(s)": " → {format}, {levels} mip(s)",
    " | correctifs : {fixes}": " | fixes: {fixes}",
    " → version locale conservée": " → local version kept",
    "== QC terminé en {seconds:.0f} s — {summary}": "== QC finished in {seconds:.0f} s — {summary}",
    "Rapport : {path}": "Report: {path}",

    # ---------------------------------------------------------------- analyse et routage
    "minuscule (table de couleurs probable)": "tiny (probably a color table)",
    "plus petite que {min_size}px": "smaller than {min_size}px",
    "couleur quasi unie": "nearly flat color",
    "dégradé lisse, sans détail à réinterpréter": "smooth gradient, no detail to reinterpret",
    "carte de relief probable (normal map)": "probable relief map (normal map)",
    "format très allongé ({aspect:.1f}:1)": "very elongated format ({aspect:.1f}:1)",
    "texture de couleur": "color texture",
    "nom « {words} »": "name \"{words}\"",
    "{flat:.0%} d'aplats": "{flat:.0%} flat areas",
    "{colors} couleurs": "{colors} colors",
    "bords francs": "hard edges",
    "feuille de {glyphs} glyphes": "sheet of {glyphs} glyphs",
    "nom « {word} »": "name \"{word}\"",
    "{green:.0%} de vert": "{green:.0%} green",
    "{skin:.0%} de teintes peau, surface lisse": "{skin:.0%} skin tones, smooth surface",
    "{axis_edges:.0%} de bords alignés": "{axis_edges:.0%} aligned edges",
    "fibres orientées ({coherence:.2f}), {brown:.0%} de brun": "oriented fibers ({coherence:.2f}), {brown:.0%} brown",
    "trame périodique ({periodic:.0f})": "periodic weave ({periodic:.0f})",
    "saturation faible ({sat:.2f})": "low saturation ({sat:.2f})",
    "aucun indice dominant": "no dominant clue",
    "donnée technique (nom « {words} »)": "technical data (name \"{words}\")",
    "graphisme plat ({kind}) : ": "flat graphics ({kind}): ",
    "peau ou visage ({why}) : agrandissement sans IA générative": "skin or face ({why}): upscale without generative AI",
    "matériau {material} ({why})": "material {material} ({why})",

    # ---------------------------------------------------------------- orchestrateur
    "fichier STOP détecté": "STOP file detected",
    "interruption clavier": "keyboard interrupt",
    "[UI] moteurs disponibles : {engines}": "[UI] available engines: {engines}",
    "aucun moteur n'a abouti": "no engine succeeded",
    "{ui} texture(s) UI traitée(s)": "{ui} UI texture(s) processed",
    "{local} texture(s) agrandie(s)": "{local} texture(s) upscaled",
    "ComfyUI injoignable : textures laissées en attente": "ComfyUI unreachable: textures left pending",
    "ComfyUI injoignable": "ComfyUI unreachable",
    "changement de workflow (Tile ↔ Tile+Canny)": "workflow change (Tile ↔ Tile+Canny)",
    "[COMFYUI] groupe : workflow {workflow}, raccord {tiling}, {n} texture(s)":
        "[COMFYUI] group: workflow {workflow}, tiling {tiling}, {n} texture(s)",
    "échecs passagers répétés : {msg}": "repeated transient failures: {msg}",
    "workflow refusé par ComfyUI, étape arrêtée : {info}": "workflow refused by ComfyUI, stage stopped: {info}",
    "plafond de requêtes payantes atteint": "paid request cap reached",
    "{n} échecs consécutifs : vérifie ComfyUI et le workflow": "{n} consecutive failures: check ComfyUI and the workflow",
    "ComfyUI injoignable : textures restantes laissées en attente": "ComfyUI unreachable: remaining textures left pending",
    "mémoire saturée (OOM) avant nouvel essai": "out of memory (OOM) before a new attempt",
    "{ai} validée(s), {fallback} repli(s), {retried} reprise(s)": "{ai} accepted, {fallback} fallback(s), {retried} retry(ies)",
    "libération de la VRAM impossible : {exc}": "could not free VRAM: {exc}",
    "{n} textures depuis la dernière libération": "{n} textures since the last free",
    "fin de lot": "end of batch",
    "VRAM libre {v:.1f} Go sous le seuil": "free VRAM {v:.1f} GB below the threshold",
    "attente du redémarrage ({seconds} s max)": "waiting for restart ({seconds} s max)",

    # ---------------------------------------------------------------- traitement (pipeline.py)
    "image de référence introuvable : {path}": "reference image not found: {path}",
    "  ! dossier introuvable : {root}": "  ! folder not found: {root}",
    "règle « {rule} »": "rule \"{rule}\"",
    "    reprise de la requête {rid}… ({name})": "    resuming request {rid}… ({name})",
    "repli {exc}": "fallback {exc}",
    "aucun résultat": "no result",
    "  ℹ {n} texture(s) traitée(s) avec d'anciens réglages (relance avec --redo-stale pour les refaire)":
        "  ℹ {n} texture(s) processed with old settings (run again with --redo-stale to redo them)",
    "  ✗ {name} : {exc}": "  ✗ {name}: {exc}",
    "  agrandissement local : {local} texture(s)": "  local upscale: {local} texture(s)",
    "  génération IA ({name}) : {total} texture(s), {workers} en parallèle":
        "  AI generation ({name}): {total} texture(s), {workers} in parallel",
    ", plafond {limit} requêtes": ", cap {limit} requests",
    "… à reprendre": "… to resume",
    "↺ repli local": "↺ local fallback",
    "✓ IA": "✓ AI",
    "■ arrêt": "■ stopped",
    "✗ erreur": "✗ error",
    "structure {structure}, essais {attempts}": "structure {structure}, attempts {attempts}",
    "  [{done}/{total}] {name} : {tag} ({detail})": "  [{done}/{total}] {name}: {tag} ({detail})",
    "  ■ interruption : les requêtes déjà lancées seront reprises au prochain run":
        "  ■ interrupted: requests already sent will be resumed on the next run",
    "arrêt demandé": "stop requested",

    # ---------------------------------------------------------------- export (postprocess.py)
    "  ! {name} illisible : {err}": "  ! {name} unreadable: {err}",
    "  {n} original(aux) sans image générée correspondante (ignorés ; --include-unmatched pour les exporter en agrandissement local)":
        "  {n} original(s) without a matching generated image (skipped; --include-unmatched to export them as local "
        "upscales)",
    "cible inconnue : {target}": "unknown target: {target}",
    "aucune image à exporter": "no image to export",
    "{mb} Mo de textures pour un budget de {budget:.0f} Mo : risque de plantage d'un jeu 32 bits (max_side ou scale à réduire)":
        "{mb} MB of textures for a {budget:.0f} MB budget: a 32-bit game may crash (reduce max_side or scale)",
    "nom en double : {file} ({a} / {b})": "duplicate name: {file} ({a} / {b})",
    "DDS : aucun résultat d'encodage": "DDS: no encoding result",
    "DDS relu invalide : {errors}": "invalid DDS on read-back: {errors}",
    "compression {fmt} : PSNR {psnr:.1f} dB < {lim:.0f} (dégradé, texte ou UI : dds_format = \"bgra8\" pour cette texture)":
        "{fmt} compression: PSNR {psnr:.1f} dB < {lim:.0f} (gradient, text or UI: dds_format = \"bgra8\" for this texture)",
    "couverture alpha des mipmaps : écart {v:.2f}": "mipmap alpha coverage: drift {v:.2f}",
    "luminosité des mipmaps : écart {v:.0%}": "mipmap brightness: drift {v:.0%}",
    "nom en double : {file}": "duplicate name: {file}",
    "{n} corrigée(s)": "{n} corrected",
    "{n} validée(s)": "{n} validated",
    "{n} locale(s)": "{n} local",
    "{n} rejetée(s)": "{n} rejected",
    "{n} mesurée(s) (choix manuel ou QC désactivé)": "{n} measured (manual choice or QC disabled)",
    "{n} erreur(s)": "{n} error(s)",
    " — DDS {mb} Mo (originaux {orig} Mo{ratio})": " — DDS {mb} MB (originals {orig} MB{ratio})",

    # ---------------------------------------------------------------- fournisseurs
    "une requête mock ne peut pas être reprise": "a mock request cannot be resumed",
    "le paquet higgsfield-client n'est pas installé : pip install -r requirements.txt":
        "the higgsfield-client package is not installed: pip install -r requirements.txt",
    "clé Higgsfield absente. Crée une clé sur console.higgsfield.ai puis définis la variable d'environnement HF_KEY=\"id:secret\" (ou HF_API_KEY et HF_API_SECRET).":
        "Higgsfield key missing. Create a key on console.higgsfield.ai, then set the environment variable "
        "HF_KEY=\"id:secret\" (or HF_API_KEY and HF_API_SECRET).",
    "accès refusé par Higgsfield ({code}) : {msg}": "access denied by Higgsfield ({code}): {msg}",
    "crédits insuffisants ({code}) : {msg}": "insufficient credits ({code}): {msg}",
    "requête refusée ({code}) : {msg}": "request refused ({code}): {msg}",
    "erreur Higgsfield ({code}) : {msg}": "Higgsfield error ({code}): {msg}",
    "envoi impossible : {exc}": "upload failed: {exc}",
    "soumission impossible : {exc}": "submission failed: {exc}",
    "réseau indisponible pendant l'attente : {exc}": "network unavailable while waiting: {exc}",
    "résultat bloqué par le filtre de contenu (NSFW)": "result blocked by the content filter (NSFW)",
    "génération en échec ({name})": "generation failed ({name})",
    "délai dépassé ; la requête sera reprise au prochain lancement": "timeout exceeded; the request will be resumed on the next run",
    "lecture du résultat impossible : {exc}": "could not read the result: {exc}",
    "réponse sans image : {data}": "response without an image: {data}",
    "téléchargement du résultat impossible : {exc}": "could not download the result: {exc}",
    "workflow ComfyUI invalide : {exc}": "invalid ComfyUI workflow: {exc}",
    "  ! nœud {title} absent du workflow : réglage ignoré": "  ! node {title} missing from the workflow: setting ignored",
    "[timeout] pas de réponse de la boucle ComfyUI : {exc}": "[timeout] no answer from the ComfyUI loop: {exc}",
    "les requêtes locales ne se reprennent pas": "local requests cannot be resumed",
    "checkpoint choisi mais aucun nœud TEXFORGE_CHECKPOINT unique dans le workflow":
        "checkpoint chosen but no single TEXFORGE_CHECKPOINT node in the workflow",
    "modèle ControlNet choisi mais aucun nœud TEXFORGE_CN_TILE_MODEL unique":
        "ControlNet model chosen but no single TEXFORGE_CN_TILE_MODEL node",
    "LoRA choisie mais aucun nœud TEXFORGE_CHECKPOINT unique pour l'insérer":
        "LoRA chosen but no single TEXFORGE_CHECKPOINT node to insert it after",

    # ---------------------------------------------------------------- contrôle qualité
    "corrigée": "corrected",
    "locale": "local",
    "rejetée": "rejected",
    "validée": "validated",
    "erreur": "error",
    "mesurée (choix manuel)": "measured (manual choice)",
    "proportions {w}x{h} ≠ original {ow}x{oh}": "proportions {w}x{h} ≠ original {ow}x{oh}",
    "résolution {w}x{h} < cible {tw}x{th} : agrandie": "resolution {w}x{h} < target {tw}x{th}: upscaled",
    "structure {v:.2f} < {lim}": "structure {v:.2f} < {lim}",
    "composition décalée ({v:.3f})": "shifted composition ({v:.3f})",
    "zone réinventée (corrélation locale {v:.2f})": "reinvented area (local correlation {v:.2f})",
    "SSIM {v:.2f} < {lim}": "SSIM {v:.2f} < {lim}",
    "ΔE moyen {v:.1f} > {lim}": "mean ΔE {v:.1f} > {lim}",
    "zones recolorées (ΔE p95 {v:.1f})": "recolored areas (ΔE p95 {v:.1f})",
    "teinte {v:+.1f}°": "hue {v:+.1f}°",
    "saturation ×{v:.2f}": "saturation ×{v:.2f}",
    "luminosité {v:+.1f} L*": "lightness {v:+.1f} L*",
    "palette changée ({v:.2f})": "palette changed ({v:.2f})",
    "couleur locale changée ({v:.0f}/255)": "local color changed ({v:.0f}/255)",
    "contraste ×{v:.2f}": "contrast ×{v:.2f}",
    "raccord {axis} {v:.2f} > {lim:.2f}": "seam {axis} {v:.2f} > {lim:.2f}",
    "raccord {axis} : bords qui ne se rejoignent pas ({v:.2f} > {lim:.2f})":
        "seam {axis}: edges that do not meet ({v:.2f} > {lim:.2f})",
    "raccord {axis} : motifs décalés de {px} px au bord": "seam {axis}: patterns offset by {px} px at the edge",
    "raccord {axis} : dérive de couleur d'un bord à l'autre (ΔE {v:.1f})":
        "seam {axis}: color drift from one edge to the other (ΔE {v:.1f})",
    "découpe alpha IoU {v:.3f}": "alpha cutout IoU {v:.3f}",
    "alpha écart {v:.3f}": "alpha deviation {v:.3f}",
    "peu de détail ajouté (×{v:.2f} vs agrandissement simple)": "little detail added (×{v:.2f} vs plain upscale)",
    "grain très fort ({v:.3f}) : bruit possible": "very strong grain ({v:.3f}): possible noise",
    "repli : {err}": "fallback: {err}",
    "le QC aurait refusé : {reasons}": "QC would have rejected: {reasons}",
    "contrôle : {reason}": "check: {reason}",
    "verrou couleur léger ×{v:.2f}": "light color lock ×{v:.2f}",
    "verrou couleur ×{v:.2f}": "color lock ×{v:.2f}",
    "fondu du raccord": "seam blend",
    "après correctif : {reason}": "after fix: {reason}",
    "raccord {values}": "seam {values}",
    "détail ×{v:.2f}": "detail ×{v:.2f}",

    # ---------------------------------------------------------------- chaîne complète (remaster.py)
    "Génération (UI, local, ComfyUI)": "Generation (UI, local, ComfyUI)",
    "Scan et routage": "Scan and routing",
    "Contrôle qualité et export DDS": "Quality control and DDS export",
    "(vide)": "(empty)",
    "dossier d'entrée introuvable : {path}": "input folder not found: {path}",
    "dossier de sortie non renseigné": "output folder not set",
    "le dossier de sortie doit être différent du dossier d'entrée": "the output folder must differ from the input folder",
    "workflow personnalisé introuvable : {path}": "custom workflow not found: {path}",
    "famille de modèle inconnue : {family}": "unknown model family: {family}",
    "format DDS inconnu : {fmt}": "unknown DDS format: {fmt}",
    "workflow refusé": "workflow refused",
    "QC : {validated} validées, {corrected} corrigées, {rejected} rejetées — {written} fichier(s) écrit(s)":
        "QC: {validated} validated, {corrected} corrected, {rejected} rejected — {written} file(s) written",
    "personnalisé ({name})": "custom ({name})",
    "arrêt demandé pendant le scan": "stop requested during the scan",
    "gardées": "kept",
    "local": "local",
    "[INFO] {seen} texture(s) analysée(s) — {new} nouvelle(s), {changed} modifiée(s), {errors} illisible(s) — routage : {routes}":
        "[INFO] {seen} texture(s) analyzed — {new} new, {changed} changed, {errors} unreadable — routing: {routes}",
    "arrêt demandé avant la génération": "stop requested before generation",
    "workflow refusé par ComfyUI ({why}) — vérifie le checkpoint, la LoRA et les nœuds personnalisés, puis relance":
        "workflow refused by ComfyUI ({why}) — check the checkpoint, the LoRA and the custom nodes, then run again",

    # ---------------------------------------------------------------- galerie (review.py)
    "Replis locaux": "Local fallbacks",
    "TEXFORGE · revue": "TEXFORGE · review",
    "TEXFORGE — Revue": "TEXFORGE — Review",
    "Validées": "Validated",
    "Erreur": "Error",
    "Erreurs": "Errors",
    "Exporter mes choix (JSON)": "Export my choices (JSON)",
    "Repli local": "Local fallback",
    "Validée": "Validated",
    "Agrandie localement": "Upscaled locally",
    "Final": "Final",
    "Original": "Original",
    "décalage": "offset",
    "raccord": "seam",
    "structure": "structure",
    "Garder le choix actuel": "Keep the current choice",
    "IA": "AI",
    "Local": "Local",
    "auto": "auto",
    "essai": "attempt",
    "Regénérer (nouvelle requête payante)": "Regenerate (new paid request)",
    "Utiliser la version IA": "Use the AI version",
    "Utiliser la version locale": "Use the local version",
    "choix modifié(s)": "choice(s) changed",
    "texture(s)": "texture(s)",

    # ---------------------------------------------------------------- agrandissement UI
    "{engine} : exécutable non configuré": "{engine}: executable not configured",
    "[UI] moteur {engine} en échec pour le lot : {exc}": "[UI] engine {engine} failed for the batch: {exc}",
    "{engine} : aucune sortie": "{engine}: no output",
    "{engine} : rejeté ({reasons})": "{engine}: rejected ({reasons})",
    "délai de {timeout:.0f} s dépassé": "{timeout:.0f} s timeout exceeded",
    "code {code} : {tail}": "code {code}: {tail}",
    "[UI] {engine} ({model}, ×{scale}) : {n} image(s) en {seconds:.1f} s":
        "[UI] {engine} ({model}, ×{scale}): {n} image(s) in {seconds:.1f} s",

    # ---------------------------------------------------------------- validation des rendus IA
    "structure {structure:.2f} < {lim}": "structure {structure:.2f} < {lim}",
    "décalage {shift:.3f} > {lim}": "offset {shift:.3f} > {lim}",
    "zone modifiée (corrélation locale {v:.2f})": "altered area (local correlation {v:.2f})",
    "raccord {v:.2f} > {lim:.2f}": "seam {v:.2f} > {lim:.2f}",
    "✗ « {name} » n'est pas texconv.exe : vide ce champ ou choisis texconv.exe.":
        '✗ "{name}" is not texconv.exe: clear this field or choose texconv.exe.',
    'Écrire les DDS dans ce dossier quand même ?':
        'Write the DDS files to this folder anyway?',
    'Special K ne charge les textures remplacées que depuis <jeu>\\SK_Res\\inject\\textures.\n\nDossier de sortie choisi :\n{out}':
        'Special K only loads replacement textures from <game>\\SK_Res\\inject\\textures.\n\nChosen output folder:\n{out}',
    '\n\nDossier attendu pour ce dump :\n{inject}':
        '\n\nExpected folder for this dump:\n{inject}',
    'Aucune texture lisible dans « {path} » : TEXFORGE lit des fichiers .dds, .png, .tga, .bmp ou .jpg séparés.':
        'No readable texture in "{path}": TEXFORGE reads separate .dds, .png, .tga, .bmp or .jpg files.',
    "Il faut d'abord les extraire pendant que le jeu tourne : lance-le avec Special K (Ctrl+Maj+Retour arrière), active le dump des textures, joue quelques minutes, puis choisis le dossier SK_Res\\dump\\textures (bouton « 🔍 Détecter »).":
        'They must first be extracted while the game runs: launch it with Special K (Ctrl+Shift+Backspace), enable texture dumping, play for a few minutes, then choose the SK_Res\\dump\\textures folder ("🔍 Detect" button).',
    'Ce jeu ({engine}) range ses textures dans des archives ({archives}).':
        'This game ({engine}) keeps its textures in archives ({archives}).',
    'Ce jeu range ses textures dans des archives ({archives}).':
        'This game keeps its textures in archives ({archives}).',
    "C'est le dossier d'installation du jeu : ses textures n'y sont pas en fichiers séparés.":
        "This is the game's install folder: its textures are not stored there as separate files.",
    "aucune texture à traiter dans le dossier d'entrée":
        'no texture to process in the input folder',
    'le champ texconv désigne « {name} », pas texconv.exe : vide-le ou choisis texconv.exe':
        'the texconv field points to "{name}", not texconv.exe: clear it or choose texconv.exe',
    'le champ Real-ESRGAN désigne « {name} », pas realesrgan-ncnn-vulkan.exe':
        'the Real-ESRGAN field points to "{name}", not realesrgan-ncnn-vulkan.exe',
    '{src_fmt} (canaux de données) non inscriptible : BGRA8 sans perte':
        '{src_fmt} (data channels) not writable: lossless BGRA8',
    " — X dans l'alpha, Y dans le vert (DXT5nm)":
        ' — X in alpha, Y in green (DXT5nm)',
    ' — X et Y dans le rouge et le vert (BC5)':
        ' — X and Y in red and green (BC5)',
    "Dossier des textures du jeu (DDS, PNG, TGA…), sous-dossiers compris.\nRempli automatiquement avec le dump Special K le plus récent. Choisir le\ndossier d'un jeu suffit : son dump Special K (à côté de l'exécutable ou dans\nson profil SKIF), sa capture RTX Remix ou le dossier qui regroupe ses\ntextures est retrouvé.":
        "Folder of the game's textures (DDS, PNG, TGA…), subfolders included.\nFilled automatically with the most recent Special K dump. Choosing a\ngame's folder is enough: its Special K dump (next to the executable or in\nits SKIF profile), its RTX Remix capture or the folder that holds its\ntextures is found.",
    '🔍 Recherche des textures dans ce dossier…':
        '🔍 Looking for textures in this folder…',
    '✓ {count} texture(s) — {what}':
        '✓ {count} texture(s) — {what}',
    '⚠ Recherche impossible : {err}':
        '⚠ Search failed: {err}',
    '[INFO] Dossier choisi : {chosen}\n   → textures : {path} ({count} fichier(s), {source})':
        '[INFO] Chosen folder: {chosen}\n   → textures: {path} ({count} file(s), {source})',
    ' (détails dans la console)':
        ' (details in the console)',
    ' (profil global SKIF)':
        ' (global SKIF profile)',
    '[INFO] Dossier de textures : {path} ({count} fichier(s))':
        '[INFO] Texture folder: {path} ({count} file(s))',
    'dump Special K':
        'Special K dump',
    'capture RTX Remix':
        'RTX Remix capture',
    'textures du jeu en fichiers séparés':
        'game textures as separate files',
    'Aucune texture lisible.':
        'No readable texture.',
    'Le dump Special K de ce jeu existe mais est vide ({path}) : active le dump des textures, joue quelques minutes, puis choisis à nouveau le dossier.':
        "This game's Special K dump exists but is empty ({path}): enable texture dumping, play for a few minutes, then choose the folder again.",
    'textures du jeu':
        'game textures',
    'Dossier introuvable : {path}':
        'Folder not found: {path}',
    'dump du même jeu':
        "same game's dump",
    "C'est un dossier d'injection Special K : la sortie de TEXFORGE, pas les textures du jeu. Choisis le dossier du jeu ou son SK_Res\\dump\\textures.":
        "This is a Special K injection folder: TEXFORGE's output, not the game's textures. Choose the game's folder or its SK_Res\\dump\\textures.",
    'aucune texture lisible dans ce dossier':
        'no readable texture in this folder',
    'dump Special K encore vide : active le dump des textures et joue':
        'Special K dump still empty: enable texture dumping and play',
    'aucune texture lisible : le jeu ({engine}) les range dans des archives — dump Special K nécessaire':
        'no readable texture: the game ({engine}) keeps them in archives — Special K dump needed',
    'aucune texture lisible : le jeu les range dans des archives — dump Special K nécessaire':
        'no readable texture: the game keeps them in archives — Special K dump needed',
    'dossier introuvable':
        'folder not found',
    "dossier d'injection (sortie de TEXFORGE), pas les textures du jeu":
        "injection folder (TEXFORGE's output), not the game's textures",
    "[INFO] Sortie {action} : elle visait le dossier d'injection d'un autre jeu":
        "[INFO] Output {action}: it pointed at another game's injection folder",
    'réglée sur {path}':
        'set to {path}',
    'vidée':
        'cleared',
    'DDS — fichiers séparés (Fallout 4, Starfield, mods)':
        'DDS — loose files (Fallout 4, Starfield, mods)',
    'vide : toutes les textures des archives':
        'empty: every texture in the archives',
    'Textures à extraire des archives BA2 (Fallout 4, Starfield), motifs séparés par « ; ».\n« *_d.dds » : textures de couleur seulement (les cartes de relief _n et spéculaires _s\nsont des données, agrandies sans IA et beaucoup plus lourdes).\nAvec un « / », le motif porte sur le chemin : « textures/architecture/*_d.dds ».':
        'Textures to extract from BA2 archives (Fallout 4, Starfield), patterns separated by ";".\n"*_d.dds": colour textures only (_n normal and _s specular maps\nare data, enlarged without AI and much heavier).\nWith a "/", the pattern applies to the path: "textures/architecture/*_d.dds".',
    'Archives BA2':
        'BA2 archives',
    '[INFO] Pour que le jeu préfère ces fichiers aux archives : dans Documents\\My Games\\<jeu>\\<jeu>Custom.ini, section [Archive], bInvalidateOlderFiles=1 et sResourceDataDirsFinal= (vide).':
        '[INFO] To make the game prefer these files over the archives: in Documents\\My Games\\<game>\\<game>Custom.ini, [Archive] section, bInvalidateOlderFiles=1 and sResourceDataDirsFinal= (empty).',
    "[INFO] Sortie : {path}\n   À la fin, copie le dossier Textures qu'il contient dans Data (ou installe ce dossier comme un mod avec Mod Organizer 2 ou Vortex).":
        '[INFO] Output: {path}\n   When done, copy the Textures folder it contains into Data (or install this folder as a mod with Mod Organizer 2 or Vortex).',
    'archives BA2 ; extraites selon le filtre ci-dessous':
        'BA2 archives; extracted according to the filter below',
    'pas une archive BA2 : {path}':
        'not a BA2 archive: {path}',
    'version BA2 {v} inconnue ({name})':
        'unknown BA2 version {v} ({name})',
    'morceau tronqué dans {name}':
        'truncated chunk in {name}',
    'archive BA2 illisible ({name}) : {exc}':
        'unreadable BA2 archive ({name}): {exc}',
    "type d'archive BA2 « {kind} » non géré ({name})":
        'unsupported BA2 archive type "{kind}" ({name})',
    '[BA2] {name} : {total} texture(s), {n} retenue(s)':
        '[BA2] {name}: {total} texture(s), {n} selected',
    '[BA2] {name} ignorée : {exc}':
        '[BA2] {name} skipped: {exc}',
    'archive BA2 corrompue ({name})':
        'corrupted BA2 archive ({name})',
    'compression BA2 {method} non gérée ({name})':
        'unsupported BA2 compression {method} ({name})',
    'archive compressée en LZ4 (Starfield) : installe le module lz4 (pip install lz4)':
        'LZ4-compressed archive (Starfield): install the lz4 module (pip install lz4)',
    '[BA2] {name} illisible : {exc}':
        '[BA2] {name} unreadable: {exc}',
    'archives BA2':
        'BA2 archives',
    'DDS en fichiers séparés':
        'DDS as loose files',
    'format {fmt} : carte de données, agrandie sans IA':
        '{fmt} format: data map, enlarged without AI',
    '[BA2] archives de textures dans {path} — filtre : {filter}':
        '[BA2] texture archives in {path} — filter: {filter}',
    '[BA2] {selected} texture(s) à traiter sur {listed} ({matched} selon le filtre ; {maxed} déjà à la taille maximale, {loose} remplacées par un fichier séparé, {cubemaps} cubemap(s) ignorées) — {written} extraite(s), {kept} déjà là':
        '[BA2] {selected} texture(s) to process out of {listed} ({matched} match the filter; {maxed} already at maximum size, {loose} replaced by a loose file, {cubemaps} cubemap(s) skipped) — {written} extracted, {kept} already there',
    'aucune texture des archives BA2 ne correspond au filtre « {filter} »':
        'no texture in the BA2 archives matches the filter "{filter}"',
    '✗ extract : indique le dossier de destination':
        '✗ extract: give the destination folder',
    '{written} texture(s) extraite(s), {kept} déjà là, {selected} retenue(s) sur {listed}':
        '{written} texture(s) extracted, {kept} already there, {selected} selected out of {listed}',
    '{n} texture(s) selon le filtre':
        '{n} texture(s) match the filter',
    "le dossier de l'exécutable du jeu":
        "the game's executable folder",
    'Unreal Engine 3 : capture Special K nécessaire (dgVoodoo2 si le jeu est en Direct3D 9)':
        'Unreal Engine 3: Special K capture needed (dgVoodoo2 if the game runs Direct3D 9)',
    '4 ou 5':
        '4 or 5',
    'Jeu Unreal Engine {generation} : ses textures sont dans des archives {container}{traits}.':
        'Unreal Engine {generation} game: its textures are in {container} archives{traits}.',
    'Unreal Engine, archives chiffrées : capture Special K en Direct3D 11 nécessaire':
        'Unreal Engine, encrypted archives: Special K capture in Direct3D 11 needed',
    'Unreal Engine : capture Special K en Direct3D 11 nécessaire':
        'Unreal Engine: Special K capture in Direct3D 11 needed',
    "Voie possible : capture en jeu avec Special K, qui ne remplace les textures qu'en Direct3D 11 :":
        'Possible route: in-game capture with Special K, which only replaces textures in Direct3D 11:',
    '1. Installe Special K pour ce jeu (SKIF, ou dxgi.dll dans {where}).':
        '1. Install Special K for this game (SKIF, or dxgi.dll in {where}).',
    '2. Force Direct3D 11 : Steam → clic droit sur le jeu → Propriétés → Options de lancement : -dx11':
        '2. Force Direct3D 11: Steam → right-click the game → Properties → Launch options: -dx11',
    '3. Active le dump des textures, joue, puis choisis à nouveau ce dossier : TEXFORGE trouvera SK_Res\\dump\\textures dans {where}.':
        '3. Enable texture dumping, play, then choose this folder again: TEXFORGE will find SK_Res\\dump\\textures in {where}.',
    'Limite : Unreal charge ses textures par morceaux pendant le jeu (streaming) ; la capture peut être partielle, à vérifier jeu par jeu.':
        'Limit: Unreal loads its textures piece by piece while playing (streaming); the capture may be partial, to be checked game by game.',
    'chiffrées (AES)':
        'encrypted (AES)',
    'compressées ({methods})':
        'compressed ({methods})',
    'signées':
        'signed',
    "TEXFORGE n'ouvre pas les archives chiffrées : il faudrait contourner la protection du jeu.":
        "TEXFORGE does not open encrypted archives: that would mean bypassing the game's protection.",
    "TEXFORGE ne lit pas encore les textures des fichiers .uasset d'Unreal.":
        'TEXFORGE does not read textures from Unreal .uasset files yet.',
    "Ce jeu embarque Direct3D 12 : s'il refuse -dx11 ou ne démarre pas ainsi, ni la capture ni le remplacement ne sont possibles.":
        'This game ships Direct3D 12: if it refuses -dx11 or does not start that way, neither capture nor replacement is possible.',
    'Jeu Unreal Engine 3 : ses textures sont rangées dans des paquets .upk et des caches .tfc, que TEXFORGE ne lit pas.':
        'Unreal Engine 3 game: its textures are stored in .upk packages and .tfc caches, which TEXFORGE does not read.',
    "Voie possible : capture en jeu avec Special K. Si le jeu est en Direct3D 9, passe-le d'abord en Direct3D 11 avec dgVoodoo2 (section 2 du README), puis active le dump des textures.":
        'Possible route: in-game capture with Special K. If the game runs Direct3D 9, first move it to Direct3D 11 with dgVoodoo2 (section 2 of the README), then enable texture dumping.',
    'Une fois le dump fait, choisis à nouveau ce dossier : TEXFORGE trouvera SK_Res\\dump\\textures dans {where}.':
        'Once the dump is done, choose this folder again: TEXFORGE will find SK_Res\\dump\\textures in {where}.',
    'Lancer quand même ?':
        'Launch anyway?',
    'Ce dossier de jeu ne contient que {n} image(s) séparée(s) : sans doute des icônes ou des fichiers annexes, pas les textures du jeu.':
        "This game folder only holds {n} separate image(s): probably icons or side files, not the game's textures.",
    '{n} image(s) annexe(s) seulement : les textures du jeu sont ailleurs — dump Special K nécessaire':
        "only {n} side image(s): the game's textures are elsewhere — Special K dump needed",
    "Ce jeu est en Direct3D {v} : Special K y gère mal les textures. Fais-le d'abord passer en Direct3D 11 avec dgVoodoo2 (section 2 du README), puis fais le dump.":
        'This game runs Direct3D {v}: Special K handles its textures poorly. First move it to Direct3D 11 with dgVoodoo2 (section 2 of the README), then dump.',
    "Ce jeu semble utiliser {api} : Special K n'y remplace pas les textures (Direct3D 11 seulement, Direct3D 9 en partie).":
        'This game seems to use {api}: Special K does not replace textures there (Direct3D 11 only, partly Direct3D 9).',
    'Ce dossier de jeu ne contient que {n} image(s) séparée(s) ({examples}) : des icônes ou des fichiers annexes, pas les textures du jeu.':
        "This game folder only holds {n} separate image(s) ({examples}): icons or side files, not the game's textures.",
    'fichier source disparu : {path}':
        'source file gone: {path}',
    'nœuds absents de ComfyUI : {nodes}{packs}. Installe-les avec ComfyUI Manager (Custom Nodes Manager) ou mets ComfyUI à jour, redémarre-le, puis relance.':
        'nodes missing from ComfyUI: {nodes}{packs}. Install them with ComfyUI Manager (Custom Nodes Manager) or update ComfyUI, restart it, then launch again.',
    'nœuds de raccord absents ({nodes}, pack ComfyUI-seamless-tiling) : TEXFORGE prolonge lui-même les textures répétables en boucle avant la génération. Raccord natif : ComfyUI Manager → Custom Nodes Manager → « ComfyUI-seamless-tiling ».':
        'seam nodes missing ({nodes}, ComfyUI-seamless-tiling pack): TEXFORGE wraps tileable textures itself before generation. Native seams: ComfyUI Manager → Custom Nodes Manager → "ComfyUI-seamless-tiling".',
    'liste des nœuds installés illisible (/object_info) : workflow envoyé tel quel ; un nœud facultatif refusé par ComfyUI sera contourné à la première texture.':
        'installed node list unreadable (/object_info): workflow sent as is; an optional node refused by ComfyUI will be bypassed on the first texture.',
    'nœuds de raccord absents de ComfyUI : contournés, texture relancée avec le raccord préparé par TEXFORGE':
        'seam nodes missing from ComfyUI: bypassed, texture requeued with seams prepared by TEXFORGE',
    '{exc} — type de nœud absent : {cls}{pack}. Installe-le avec ComfyUI Manager (Custom Nodes Manager) ou mets ComfyUI à jour, redémarre-le, puis relance.':
        '{exc} — missing node type: {cls}{pack}. Install it with ComfyUI Manager (Custom Nodes Manager) or update ComfyUI, restart it, then run again.',
    '[INFO] TEXFORGE {version} lancé depuis {program}':
        '[INFO] TEXFORGE {version} running from {program}',
    '[LOCAL] {n} texture(s) agrandie(s) par Lanczos faute de Real-ESRGAN : refaite(s) avec Real-ESRGAN':
        '[LOCAL] {n} texture(s) upscaled with Lanczos because Real-ESRGAN was missing: redone with Real-ESRGAN',
    '[INFO] rien de nouveau à générer : {n} texture(s) déjà traitée(s) lors d\'un lancement précédent, exportée(s) telles quelles':
        '[INFO] nothing new to generate: {n} texture(s) already processed in a previous run, exported as they are',
    '[INFO] {n} texture(s) de peau ou de visage : agrandie(s) sans IA générative (l\'IA redessinerait les traits)':
        '[INFO] {n} skin or face texture(s): upscaled without generative AI (the AI would redraw the features)',
    '◧ Avant / Après':
        '◧ Before / After',
    'Afficher ou masquer le panneau avant / après (original à gauche, IA à droite)':
        'Show or hide the before / after panel (original on the left, AI on the right)',
    "Tout passer par l'IA (ComfyUI)":
        'Send everything through AI (ComfyUI)',
    "Peau, visages, HUD, aplats et petites textures partent aussi vers ComfyUI.\nRestent hors IA : normal maps et cartes de données (l'IA en ferait des données\nfausses) et polices (lettres déformées).":
        'Skin, faces, HUD, flat colours and small textures also go to ComfyUI.\nKept out of AI: normal maps and data maps (the AI would turn them into wrong\ndata) and fonts (distorted letters).',
    'IA par morceaux à la taille finale':
        'AI in tiles at the final size',
    'Le modèle travaille à environ 1 mégapixel. Coché : la texture est agrandie, puis\ngénérée morceau par morceau à sa taille finale (vrai détail en 2048 / 4096).\nDécoché : une seule passe à 1 mégapixel, puis agrandissement classique.':
        'The model works at about 1 megapixel. Checked: the texture is upscaled, then\ngenerated tile by tile at its final size (real detail at 2048 / 4096).\nUnchecked: a single pass at 1 megapixel, then a plain upscale.',
    'Peau, visages et HUD compris. Restent hors IA : normal maps, cartes de données et polices.':
        'Skin, faces and HUD included. Kept out of AI: normal maps, data maps and fonts.',
    'Vrai détail IA en 2048 / 4096 : 9 passages par texture en 2048, 25 en 4096 (plusieurs minutes par texture en 4096).':
        'Real AI detail at 2048 / 4096: 9 passes per 2048 texture, 25 per 4096 (several minutes per 4096 texture).',
    'QC : {label}':
        'QC: {label}',
    'détail ×{gain:.2f}':
        'detail ×{gain:.2f}',
    'Real-ESRGAN (mode réaliste)':
        'Real-ESRGAN (realistic mode)',
    'Essai IA écarté — le fichier exporté est la version locale':
        'AI attempt discarded — the exported file is the local version',
    'matériau {material}':
        'material {material}',
    'essai {n}':
        'attempt {n}',
    'raison : {why}':
        'reason: {why}',
    'IA (ComfyUI)':
        'AI (ComfyUI)',
    'routage : {why}':
        'routing: {why}',
    'Agrandissement UI ({engine}), sans IA générative':
        'UI upscale ({engine}), no generative AI',
    'Agrandissement sans IA ({engine})':
        'Upscale without AI ({engine})',
    '{stem} : morceau {i}/{n}':
        '{stem}: tile {i}/{n}',
    "tout par l'IA : matériau {material} ({why})":
        'everything through AI: material {material} ({why})',
    'police de caractères : gardée hors IA (lettres déformées sinon)':
        'font: kept out of AI (letters would be distorted)',
    'AVANT / APRÈS':
        'BEFORE / AFTER',
    'Texture précédente':
        'Previous texture',
    'Texture suivante':
        'Next texture',
    'IA seulement':
        'AI only',
    "Ne lister que les textures passées par l'IA":
        'List only textures that went through AI',
    'Suivre':
        'Follow',
    "Pendant un lancement, afficher chaque texture dès qu'elle est terminée":
        'During a run, show each texture as soon as it is finished',
    'Résultat':
        'Result',
    'Ajuster':
        'Fit',
    'Image entière (double-clic sur une image)':
        'Whole image (double-click an image)',
    "Un pixel du résultat = un pixel de l'écran":
        'One result pixel = one screen pixel',
    "Aucune texture traitée dans ce projet pour l'instant. Lance le pipeline : chaque texture apparaîtra ici dès qu'elle est terminée.":
        'No texture processed in this project yet. Run the pipeline: each texture will appear here as soon as it is finished.',
    'Original — {w}×{h}':
        'Original — {w}×{h}',
    '{label} — chargement…':
        '{label} — loading…',
    '{label} — {w}×{h}':
        '{label} — {w}×{h}',
    '{label} — image illisible':
        '{label} — unreadable image',
    'Original illisible : {err}':
        'Original unreadable: {err}',
    'Z-Image Turbo — ControlNet Tile (model patch) si présent':
        'Z-Image Turbo — Tile ControlNet (model patch) when present',
    "[INFO] ComfyUI n'a aucun checkpoint SDXL / SD 1.5 mais a Z-Image Turbo : famille Z-Image Turbo choisie.":
        '[INFO] ComfyUI has no SDXL / SD 1.5 checkpoint but has Z-Image Turbo: Z-Image Turbo family selected.',
    'denoise / ControlNet Tile effectifs : {txt} — sans le fichier du ControlNet Tile : denoise ×{f}':
        'effective denoise / Tile ControlNet: {txt} — without the Tile ControlNet file: denoise ×{f}',
    '[INFO] Z-Image Turbo absent de ComfyUI : famille SDXL choisie.':
        '[INFO] Z-Image Turbo missing from ComfyUI: SDXL family selected.',
    "n'importe quel checkpoint SDXL convient, ex. huggingface.co/RunDiffusion/Juggernaut-XL-v9 (7,1 Go)":
        'any SDXL checkpoint will do, e.g. huggingface.co/RunDiffusion/Juggernaut-XL-v9 (7.1 GB)',
    'huggingface.co/xinsir/controlnet-tile-sdxl-1.0 (diffusion_pytorch_model.safetensors, 2,5 Go, à renommer ainsi)':
        'huggingface.co/xinsir/controlnet-tile-sdxl-1.0 (diffusion_pytorch_model.safetensors, 2.5 GB, rename it to this)',
    'huggingface.co/xinsir/controlnet-union-sdxl-1.0 (diffusion_pytorch_model_promax.safetensors, 2,5 Go, à renommer ainsi)':
        'huggingface.co/xinsir/controlnet-union-sdxl-1.0 (diffusion_pytorch_model_promax.safetensors, 2.5 GB, rename it to this)',
    "n'importe quel checkpoint SD 1.5 convient":
        'any SD 1.5 checkpoint will do',
    'huggingface.co/lllyasviel/ControlNet-v1-1':
        'huggingface.co/lllyasviel/ControlNet-v1-1',
    'huggingface.co/Comfy-Org/z_image_turbo':
        'huggingface.co/Comfy-Org/z_image_turbo',
    'dossier vide':
        'empty folder',
    'présents : {names}':
        'present: {names}',
    "ton ComfyUI n'a pas de checkpoint pour ce workflow mais il a Z-Image Turbo ({name}) : choisis la famille « Z-Image Turbo » dans TEXFORGE, sans rien télécharger. ":
        'your ComfyUI has no checkpoint for this workflow but it has Z-Image Turbo ({name}): choose the "Z-Image Turbo" family in TEXFORGE, no download needed. ',
    'modèles absents de ComfyUI : {items}. Après un ajout, redémarre ComfyUI puis relance.':
        'models missing from ComfyUI: {items}. After adding them, restart ComfyUI and run again.',
    'modèle « {val} » absent de ComfyUI : « {pick} » utilisé à la place.':
        'model "{val}" missing from ComfyUI: "{pick}" used instead.',
    '« {val} » (models\\{folder}, {have})':
        '"{val}" (models\\{folder}, {have})',
    'workflow Tile+Canny désactivé : modèle « {val} » absent de ComfyUI (dossier models\\{folder}) ; toutes les matières passent par le workflow Tile.':
        'Tile+Canny workflow disabled: model "{val}" missing from ComfyUI (models\\{folder} folder); all materials use the Tile workflow.',
    "ControlNet Tile « {val} » absent de ComfyUI (models\\model_patches) : img2img sans ControlNet, denoise réduit. Avec lui, l'IA peut ajouter bien plus de détail sans redessiner les formes.":
        'Tile ControlNet "{val}" missing from ComfyUI (models\\model_patches): img2img without ControlNet, reduced denoise. With it, the AI can add much more detail without redrawing shapes.',
    "[INFO] ⚠ Real-ESRGAN introuvable : les morceaux partiront d'un agrandissement Lanczos flou et l'IA ajoutera peu de détail. Bouton « Installer Real-ESRGAN ».":
        '[INFO] ⚠ Real-ESRGAN not found: tiles will start from a blurry Lanczos upscale and the AI will add little detail. Use "Install Real-ESRGAN".',
    '[INFO] Real-ESRGAN réglé introuvable ({old}) : {new} utilisé.':
        '[INFO] Configured Real-ESRGAN not found ({old}): {new} used.',
    'Fidèle — textures de jeu, DA conservée':
        'Faithful — game textures, art direction kept',
    'Photoréaliste créatif — invente du détail':
        'Creative photoreal — invents detail',
    "Fidèle : textures de jeu, la DA et les formes sont conservées, le contrôle qualité rejette ce qui s'en écarte.\nPhotoréaliste créatif : le modèle voit l'image entière, invente matière et détail (denoise jusqu'à 1),\npuis affine morceau par morceau ; le contrôle qualité ne garde qu'un garde-fou.":
        'Faithful: game textures, art direction and shapes are kept, quality control rejects what drifts.\nCreative photoreal: the model sees the whole image, invents material and detail (denoise up to 1),\nthen refines tile by tile; quality control only keeps a safety net.',
    'Rendu':
        'Rendering',
    "Part de l'image réinventée par le modèle. Bas : fidèle à l'original ;\nhaut : plus de matière et de détail, plus de risque de trahir la DA.\nAu-delà de 0,6 : rendu « Photoréaliste créatif » conseillé.":
        'Share of the image reinvented by the model. Low: faithful to the original;\nhigh: more material and detail, more risk of betraying the art direction.\nAbove 0.6: "Creative photoreal" rendering recommended.',
    '⚠ Au-delà de 0,6 en rendu Fidèle, le contrôle qualité rejettera la plupart des résultats (version locale gardée) : choisis « Photoréaliste créatif ».':
        '⚠ Above 0.6 in Faithful rendering, quality control will reject most results (local version kept): choose "Creative photoreal".',
    "passe globale : denoise {d:.2f} sur l'image entière, puis affinage par morceaux à {r:.2f} ; ControlNet Tile {c:.2f}":
        'global pass: denoise {d:.2f} on the whole image, then tile refinement at {r:.2f}; Tile ControlNet {c:.2f}',
    '{stem} : passe globale':
        '{stem}: global pass',
    'rendu inconnu : {style}':
        'unknown rendering: {style}',
    'Denoise auto (maximum)':
        'Auto denoise (maximum)',
    "Fidèle : part de 0,60, le plus fort denoise qui garde d'ordinaire la texture intacte, et redescend\n(0,49 puis 0,41) seulement si la validation voit la texture transformée.\nPhotoréaliste créatif : part de 1,00 (puis 0,85 et 0,70 si le garde-fou échoue).":
        'Faithful: starts at 0.60, the strongest denoise that usually keeps the texture intact, and steps down\n(0.49 then 0.41) only if validation sees the texture transformed.\nCreative photoreal: starts at 1.00 (then 0.85 and 0.70 if the safety net fails).',
    '(si refus : {steps})':
        '(if rejected: {steps})',
    '{stem} : essai du denoise {d:.2f}':
        '{stem}: trying denoise {d:.2f}',
    "Décris le résultat voulu plutôt qu'une consigne, en anglais de préférence.\nEx. : green futuristic armor, scratched metal plates, worn paint, gold reflective visor":
        'Describe the result you want rather than an instruction, preferably in English.\nE.g.: green futuristic armor, scratched metal plates, worn paint, gold reflective visor',
    "Ajouté au prompt de chaque texture du lancement. Le modèle génère une image qui correspond à une description :\n« rusty scratched metal » marche mieux que « ajoute de la rouille ». SDXL comprend surtout l'anglais ;\nZ-Image comprend aussi le français. Une règle par texture ([[rules]] hint) s'y ajoute.":
        'Added to the prompt of every texture in the run. The model generates an image that matches a description:\n"rusty scratched metal" works better than "add rust". SDXL mostly understands English;\nZ-Image also understands French. A per-texture rule ([[rules]] hint) is added to it.',
    'Prompt à donner':
        'Prompt to give',
    'Prompt seul, sans ajouts':
        'Prompt alone, no additions',
    "Décoché : TEXFORGE l'entoure de son préfixe, du prompt du matériau détecté (rendu Fidèle)\net de son suffixe. Coché : seul ton texte part, tu as la main sur tout.":
        'Unchecked: TEXFORGE wraps it with its prefix, the detected material prompt (Faithful rendering)\nand its suffix. Checked: only your text is sent, you control everything.',
    'Prompt envoyé (texture générique) : {p}':
        'Prompt sent (generic texture): {p}',
    'nœud ColorTransfer absent : couleurs recalées par TEXFORGE après la génération.':
        'ColorTransfer node missing: colours realigned by TEXFORGE after generation.',
    'nœud OffsetImage absent : aperçu du raccord dans ComfyUI désactivé (sans effet sur le résultat).':
        'OffsetImage node missing: seam preview in ComfyUI disabled (no effect on the result).',
    'workflow Tile+Canny désactivé, nœuds absents : {nodes} ; toutes les matières passent par le workflow Tile.':
        'Tile+Canny workflow disabled, missing nodes: {nodes}; every material goes through the Tile workflow.',
    "[INFO] {n} texture(s) d'un scan précédent introuvable(s) (fichier supprimé ou renommé) : mise(s) de côté":
        '[INFO] {n} texture(s) from a previous scan not found (file deleted or renamed): set aside',
    "Durée du lancement entier (analyse, génération, contrôle qualité et export).\nCalculée sur le nombre de passages ComfyUI et la taille des textures, puis recalée sur le temps réellement mis par les textures déjà terminées.\nLes durées mesurées sont mémorisées : l'estimation du lancement suivant sur ce PC part d'elles.":
        'Duration of the whole run (scan, generation, quality control and export).\nComputed from the number of ComfyUI passes and the size of the textures, then readjusted to the time the textures already finished actually took.\nMeasured durations are remembered: the estimate of the next run on this PC starts from them.',
    "Durée totale estimée : calculée à la fin de l'analyse des textures":
        'Estimated total duration: computed once the textures are scanned',
    'Durée totale estimée ~{total} · reste ~{rest} · fin vers {when}':
        'Estimated total duration ~{total} · ~{rest} left · ends around {when}',
    'Durée réelle {real} — estimation de départ ~{first}':
        'Actual duration {real} — initial estimate ~{first}',
    '{d} j {h} h':
        '{d} d {h} h',
    '{s} s':
        '{s} s',
    '{m} min':
        '{m} min',
    '{h} h {m:02d}':
        '{h} h {m:02d}',
    ' — recalée pendant le traitement':
        ' — readjusted during processing',
    'mesuré sur ce PC':
        'measured on this PC',
    'valeur indicative, mesurée pendant ce lancement':
        'rough default, measured during this run',
    'génération ~{gen} : {passes} passage(s) ComfyUI × ~{per} s ({src})':
        'generation ~{gen}: {passes} ComfyUI pass(es) × ~{per} s ({src})',
    'contrôle qualité et export ~{t} ({n} texture(s), {mp} Mpx)':
        'quality control and export ~{t} ({n} texture(s), {mp} MP)',
    '[ESTIMATION] ≈ {total} au total':
        '[ESTIMATE] ≈ {total} in total',
    'génération ~{gen}':
        'generation ~{gen}',
    '[ESTIMATION] indisponible : {exc}':
        '[ESTIMATE] unavailable: {exc}',
    'Installer TEXFORGE {remote} depuis GitHub ({repo}) ?\n\nVersion installée : {local}.':
        'Install TEXFORGE {remote} from GitHub ({repo})?\n\nInstalled version: {local}.',
    '⬇ Mettre à jour':
        '⬇ Update',
    'Mises à jour : vérification au démarrage…':
        'Updates: checking at startup…',
    'Installation de TEXFORGE {remote} en cours…':
        'Installing TEXFORGE {remote}…',
    'Vérifier maintenant':
        'Check now',
    'Ouvrir le dépôt sur GitHub ({repo})':
        'Open the repository on GitHub ({repo})',
    'Vérifier automatiquement (démarrage, puis toutes les 2 min)':
        'Check automatically (at startup, then every 2 min)',
    "La nouvelle version est téléchargée puis contrôlée avant toute écriture ; les fichiers remplacés sont sauvegardés (retour arrière possible depuis la pastille). L'application redémarre ensuite.":
        'The new version is downloaded and checked before anything is written; replaced files are backed up (roll back from the badge). The application then restarts.',
    'vérification impossible : {why}':
        'check failed: {why}',
    '● installation…':
        '● installing…',
    '● {v} installée':
        '● {v} installed',
    "TEXFORGE {v} est installée sur le disque ; la version {running} tourne encore.\nRedémarre l'application pour l'utiliser.":
        'TEXFORGE {v} is installed on disk; version {running} is still running.\nRestart the application to use it.',
    'Dernière vérification : {t}':
        'Last check: {t}',
    'Clic : vérifier maintenant, dépôt GitHub, retour arrière.':
        'Click: check now, GitHub repository, roll back.',
    'Possible après le traitement en cours.':
        'Available once the current run is over.',
    'Revenir à la version {v} (avant la dernière mise à jour)':
        'Go back to version {v} (before the last update)',
    'Dernier changement : {msg}':
        'Latest change: {msg}',
    "[MAJ] TEXFORGE {to} installée ({n} fichier(s) remplacé(s), {a} ajouté(s)) — redémarre l'application pour l'utiliser.":
        '[UPDATE] TEXFORGE {to} installed ({n} file(s) replaced, {a} added) — restart the application to use it.',
    "Revenir à TEXFORGE {v} ?\n\nLes fichiers d'avant la dernière mise à jour ({to}) sont remis en place, puis l'application redémarre.":
        'Go back to TEXFORGE {v}?\n\nThe files from before the last update ({to}) are put back, then the application restarts.',
    '[MAJ] Retour à TEXFORGE {to} : {n} fichier(s) restauré(s), {a} retiré(s).':
        '[UPDATE] Back to TEXFORGE {to}: {n} file(s) restored, {a} removed.',
    'Redémarrage possible après le traitement en cours.':
        'Restart possible once the current run is over.',
    '↻ Redémarrer':
        '↻ Restart',
    '● {v} disponible':
        '● {v} available',
    'TEXFORGE {remote} disponible sur GitHub ({repo}) — installée : {local}':
        'TEXFORGE {remote} available on GitHub ({repo}) — installed: {local}',
    '[MAJ] ✗ mise à jour impossible : {err}':
        '[UPDATE] ✗ update failed: {err}',
    'Mise à jour impossible :\n\n{err}':
        'Update failed:\n\n{err}',
    'TEXFORGE {to} est installée.\n\nRedémarrer maintenant ?':
        'TEXFORGE {to} is installed.\n\nRestart now?',
    '[MAJ] ✗ retour arrière impossible : {err}':
        '[UPDATE] ✗ rollback failed: {err}',
    'Retour arrière impossible :\n\n{err}':
        'Rollback failed:\n\n{err}',
    'Redémarrage impossible : {err}\nRelance TEXFORGE à la main.':
        'Restart failed: {err}\nStart TEXFORGE again by hand.',
    '[MAJ] TEXFORGE {remote} disponible sur GitHub (installée : {local}) — bouton « Mettre à jour » en haut à droite.':
        '[UPDATE] TEXFORGE {remote} available on GitHub (installed: {local}) — "Update" button at the top right.',
    "Recherche d'une mise à jour sur GitHub…":
        'Looking for an update on GitHub…',
    '● téléchargement {pct} %':
        '● downloading {pct} %',
    '● téléchargement {mb:.1f} Mo':
        '● downloading {mb:.1f} MB',
    'TEXFORGE {to} est remise en place.\n\nRedémarrer maintenant ?':
        'TEXFORGE {to} is back in place.\n\nRestart now?',
    '[MAJ] TEXFORGE {local} : plus récente que la version du dépôt GitHub ({remote}), rien à installer.':
        '[UPDATE] TEXFORGE {local}: newer than the GitHub repository version ({remote}), nothing to install.',
    '[MAJ] TEXFORGE {local} est à jour (dépôt GitHub : {remote}).':
        '[UPDATE] TEXFORGE {local} is up to date (GitHub repository: {remote}).',
    'Installation suivie par git : mets-la à jour avec « git pull ».':
        'Install tracked by git: update it with "git pull".',
    '{err}\nNouvel essai automatique dans 2 min ; clic sur la pastille pour réessayer.':
        '{err}\nAutomatic retry in 2 min; click the badge to try again.',
    'TEXFORGE {local} : plus récente que la version du dépôt GitHub ({remote}).\nRien à installer ; publie-la sur GitHub pour que les autres installations la reçoivent.':
        'TEXFORGE {local}: newer than the GitHub repository version ({remote}).\nNothing to install; publish it on GitHub so other installs receive it.',
    'TEXFORGE {local} est à jour (dépôt GitHub {repo}).':
        'TEXFORGE {local} is up to date (GitHub repository {repo}).',
    'Version installée : {local} — dépôt {repo} : {remote}':
        'Installed version: {local} — repository {repo}: {remote}',
    "TEXFORGE {to} installé : relance l'application (retour arrière : texforge update --rollback).":
        'TEXFORGE {to} installed: restart the application (roll back: texforge update --rollback).',
    'Retour à TEXFORGE {to} : {n} fichier(s) restauré(s), {a} retiré(s).':
        'Back to TEXFORGE {to}: {n} file(s) restored, {a} removed.',
    'Mise à jour disponible : {remote}':
        'Update available: {remote}',
    'Version installée plus récente que le dépôt : rien à installer.':
        'Installed version newer than the repository: nothing to install.',
    'Aucune mise à jour.':
        'No update.',
    '  téléchargé : {mb:.1f} Mo':
        '  downloaded: {mb:.1f} MB',
    'Installer la version {remote} ? [o/N] ':
        'Install version {remote}? [y/N] ',
    'version introuvable dans le dépôt {repo}':
        'version not found in repository {repo}',
    '[MAJ] Dépendances à jour.':
        '[UPDATE] Dependencies up to date.',
    '[MAJ] {n} fichier(s) remplacé(s), {a} ajouté(s), {u} identique(s) ; originaux sauvegardés dans {path}':
        '[UPDATE] {n} file(s) replaced, {a} added, {u} identical; originals backed up in {path}',
    'GitHub limite les requêtes, nouvel essai plus tard':
        'GitHub is rate-limiting requests, will retry later',
    'dépôt {repo} introuvable ou privé':
        'repository {repo} not found or private',
    'archive du dépôt vide':
        'repository archive is empty',
    "dépôt incomplet, fichier(s) absent(s) : {files} — rien n'a été modifié":
        'incomplete repository, missing file(s): {files} — nothing was changed',
    "l'archive contient la version {got} au lieu de {want} — rien n'a été modifié":
        'the archive contains version {got} instead of {want} — nothing was changed',
    "dépôt incomplet, module(s) absent(s) : {list} — rien n'a été modifié. Publie tous les fichiers de la version sur GitHub.":
        'incomplete repository, missing module(s): {list} — nothing was changed. Publish every file of the version on GitHub.',
    'installation des dépendances impossible (pip, code {code}) : {tail}\nÀ lancer à la main : {cmd}':
        'could not install dependencies (pip, code {code}): {tail}\nRun by hand: {cmd}',
    "la nouvelle version ne démarre pas (import d'essai) : {err} — rien n'a été modifié":
        'the new version does not start (test import): {err} — nothing was changed',
    'installation suivie par git ({root}) : mets-la à jour avec « git pull »':
        'install tracked by git ({root}): update it with "git pull"',
    'aucune version plus récente à installer (installée {local}, dépôt {remote})':
        'no newer version to install (installed {local}, repository {remote})',
    '[MAJ] Téléchargement de TEXFORGE {v} depuis GitHub ({repo})…':
        '[UPDATE] Downloading TEXFORGE {v} from GitHub ({repo})…',
    '[MAJ] Contrôle de la nouvelle version ({n} fichiers)…':
        '[UPDATE] Checking the new version ({n} files)…',
    "[MAJ] Import d'essai de la nouvelle version…":
        '[UPDATE] Test import of the new version…',
    'aucune sauvegarde de mise à jour à restaurer':
        'no update backup to restore',
    "la version installée ({cur}) n'est plus celle de la dernière mise à jour ({to}) : retour arrière annulé":
        'the installed version ({cur}) is no longer the one from the last update ({to}): rollback cancelled',
    'téléchargement impossible : {why}':
        'download failed: {why}',
    'archive du dépôt illisible : {err}':
        'unreadable repository archive: {err}',
    'installation des dépendances impossible : {err}':
        'could not install dependencies: {err}',
    "import d'essai impossible : {err} — rien n'a été modifié":
        'test import failed: {err} — nothing was changed',
    "écriture impossible ({err}) : fichiers d'origine restaurés":
        'write failed ({err}): original files restored',
    '[MAJ] Dépendances modifiées : installation avec pip…':
        '[UPDATE] Dependencies changed: installing with pip…',
    'restauration incomplète : {err} (sauvegarde dans {path})':
        'incomplete restore: {err} (backup in {path})',
    "chemin refusé dans l'archive : {name}":
        'path refused in the archive: {name}',
    "{path} existe déjà et n'est pas un fichier : rien n'a été modifié":
        '{path} already exists and is not a file: nothing was changed',
    'archive du dépôt trop grosse (plus de {mb} Mo) : mise à jour annulée':
        'repository archive too large (over {mb} MB): update cancelled',
    "erreur de syntaxe dans {file} (ligne {line}) : version du dépôt cassée, rien n'a été modifié":
        'syntax error in {file} (line {line}): repository version is broken, nothing was changed',
    'importé par {file}':
        'imported by {file}',
    'écriture impossible ({err}) et restauration incomplète ({err2}) : sauvegarde dans {path}':
        'write failed ({err}) and incomplete restore ({err2}): backup in {path}',
    'Prompt détaillé en anglais — aperçu':
        'Detailed English prompt — preview',
    'Ton texte :':
        'Your text:',
    '↻ Régénérer':
        '↻ Regenerate',
    'Utiliser ce prompt':
        'Use this prompt',
    'Génération en cours…':
        'Generating…',
    '✨ Détailler en anglais…':
        '✨ Detail in English…',
    "Propose un prompt beaucoup plus détaillé, en anglais, à partir de ton texte : aperçu modifiable avant de l'utiliser.\nModèle de langage local : Qwen3-4B de Z-Image dans ComfyUI (nœud TextGenerate), sinon LM Studio ou Ollama.\nLongueur adaptée au modèle : courte pour SDXL / SD 1.5, longue pour Z-Image.":
        "Suggests a much more detailed English prompt from your text: editable preview before you use it.\nLocal language model: Z-Image's Qwen3-4B in ComfyUI (TextGenerate node), otherwise LM Studio or Ollama.\nLength fitted to the model: short for SDXL / SD 1.5, long for Z-Image.",
    "Prompt proposé (modifiable avant de l'utiliser) :":
        'Suggested prompt (editable before use):',
    '{engine} · {words} mots · {s} s':
        '{engine} · {words} words · {s} s',
    '[PROMPT] prompt détaillé en anglais appliqué ({n} mots) — Ctrl+Z dans la zone pour revenir à ton texte':
        '[PROMPT] detailed English prompt applied ({n} words) — Ctrl+Z in the box to get your text back',
    '{model} dans ComfyUI : génération du prompt (le premier appel charge le modèle)…':
        '{model} in ComfyUI: generating the prompt (the first call loads the model)…',
    'pas de réponse de ComfyUI en {s} s':
        'no answer from ComfyUI within {s} s',
    '{name} ({model}) : génération du prompt…':
        '{name} ({model}): generating the prompt…',
    "aucun modèle de langage disponible pour détailler le prompt :\n- {why}\n\nIl faut soit ComfyUI à jour avec l'encodeur Qwen3-4B de Z-Image Turbo (qwen_3_4b.safetensors), soit LM Studio ou Ollama lancés avec un modèle.":
        "no language model available to detail the prompt:\n- {why}\n\nYou need either an up-to-date ComfyUI with Z-Image Turbo's Qwen3-4B text encoder (qwen_3_4b.safetensors), or LM Studio or Ollama running with a model.",
    'ComfyUI sans le nœud TextGenerate (version trop ancienne : mets ComfyUI à jour)':
        'ComfyUI without the TextGenerate node (version too old: update ComfyUI)',
    'aucun encodeur de texte Qwen dans ComfyUI (models\\text_encoders) ; celui de Z-Image Turbo, qwen_3_4b.safetensors, convient':
        "no Qwen text encoder in ComfyUI (models\\text_encoders); Z-Image Turbo's, qwen_3_4b.safetensors, works",
    'ComfyUI a refusé la demande : {err}':
        'ComfyUI refused the request: {err}',
    '{name} sans modèle':
        '{name} has no model',
    "écris d'abord quelques mots dans « Prompt à donner »":
        'write a few words in "Prompt to give" first',
    'ComfyUI injoignable ({err})':
        'ComfyUI unreachable ({err})',
    '{name} non lancé':
        '{name} not running',
    '{name} : échec ({err})':
        '{name}: failed ({err})',
    '{engine} : réponse vide':
        '{engine}: empty answer',
    'échec dans ComfyUI : {err}':
        'failure in ComfyUI: {err}',
    '{model} (ComfyUI)':
        '{model} (ComfyUI)',
    '[PUBLICATION] ✗ {err}':
        '[PUBLISH] ✗ {err}',
    'morceau démesuré dans {name}':
        'oversized chunk in {name}',
    "{n} fichier(s) de la version {v} absent(s) ou différent(s) de la liste publiée ({files}) — rien n'a été modifié. Envoi incomplet sur GitHub (envoie tout le contenu du dossier de la version), ou fichiers modifiés après la publication.":
        '{n} file(s) of version {v} missing or different from the published list ({files}) — nothing was changed. Incomplete upload to GitHub (upload the whole content of the version folder), or files changed after publishing.',
    'archive du dépôt anormale (plus de {n} fichiers ou de {mb} Mo une fois décompressée) : mise à jour annulée':
        'abnormal repository archive (more than {n} files or {mb} MB once decompressed): update cancelled',
    'liste des fichiers de la version illisible ({file})':
        'unreadable version file list ({file})',
    "requirements.txt du dépôt refusé, ligne non standard : « {line} » — rien n'a été modifié":
        'repository requirements.txt refused, non-standard line: "{line}" — nothing was changed',
    'Préparer le dossier à publier sur GitHub…':
        'Prepare the folder to publish on GitHub…',
    '[PUBLICATION] TEXFORGE {v} : {n} fichiers dans {dest}':
        '[PUBLISH] TEXFORGE {v}: {n} files in {dest}',
    "Dossier prêt : {dest}\n\nEnvoie son CONTENU sur GitHub (Add file → Upload files, glisse tous les fichiers et dossiers). Ton README sur GitHub n'est pas touché.":
        'Folder ready: {dest}\n\nUpload its CONTENT to GitHub (Add file → Upload files, drag every file and folder). Your README on GitHub is left untouched.',
    'TEXFORGE {v} : {n} fichiers dans {dest}\nEnvoie le CONTENU de ce dossier sur GitHub.':
        'TEXFORGE {v}: {n} files in {dest}\nUpload the CONTENT of this folder to GitHub.',
}
