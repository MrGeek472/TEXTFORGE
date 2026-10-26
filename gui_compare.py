"""Panneau « Avant / Après » de l'interface TEXFORGE : l'original à gauche, le résultat (IA) à droite,
zoom et déplacement synchronisés. Les images se chargent hors du fil de l'interface."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                               QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from texforge.compare import CompareItem, list_items, load_rgba
from texforge.i18n import tr

KIND_COLORS = {"ai": "#3fcf8e", "esrgan": "#45d0db", "ai_rejected": "#f2c14e", "local": "#8a93a3", "ui": "#b38cff"}
MIN_ZOOM, MAX_ZOOM = 0.02, 32.0


def load_qimage(path: str) -> QImage:
    """QImage RGBA (DDS compris) ; appelé dans un fil de fond (QImage, pas QPixmap)."""
    if Path(path).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        img = QImage(path)
        if not img.isNull():
            return img.convertToFormat(QImage.Format_RGBA8888)
    arr = load_rgba(path)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


class SyncView(QGraphicsView):
    """Vue d'une image : molette = zoom sous le curseur, glisser = déplacer, double-clic = ajuster."""
    moved = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#0f1115"))
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumSize(160, 160)
        self.horizontalScrollBar().valueChanged.connect(lambda _v: self.moved.emit())
        self.verticalScrollBar().valueChanged.connect(lambda _v: self.moved.emit())

    def set_image(self, pix: QPixmap | None, scene_size: tuple[int, int]) -> None:
        """L'image occupe toujours le rectangle de la scène (taille du résultat) : un pixel de l'original
        couvre k pixels de la scène, les deux vues restent superposables."""
        W, H = scene_size
        self.scene().setSceneRect(QRectF(0, 0, W, H))
        if pix is None or pix.isNull():
            self.item.setPixmap(QPixmap())
            return
        self.item.setPixmap(pix)
        self.item.setTransform(QTransform.fromScale(W / pix.width(), H / pix.height()))
        self._update_filter()

    def zoom(self) -> float:
        return self.transform().m11()

    def _update_filter(self) -> None:
        # agrandi à l'écran : pixels nets (on voit ce qui est réellement dans la texture) ; réduit : lissé
        sx = self.zoom() * self.item.transform().m11()
        self.item.setTransformationMode(Qt.FastTransformation if sx >= 1.0 else Qt.SmoothTransformation)

    def set_zoom(self, z: float) -> None:
        z = max(MIN_ZOOM, min(MAX_ZOOM, z))
        self.setTransform(QTransform.fromScale(z, z))
        self._update_filter()
        self.moved.emit()

    def fit(self) -> None:
        r = self.sceneRect()
        if r.isEmpty():
            return
        self.fitInView(r, Qt.KeepAspectRatio)
        self._update_filter()
        self.moved.emit()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        z = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom() * (1.25 ** steps)))
        f = z / self.zoom()
        self.scale(f, f)
        self._update_filter()
        self.moved.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()

    def center_point(self) -> QPointF:
        return self.mapToScene(self.viewport().rect().center())


class ImagePane(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.caption = QLabel(title)
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("font-weight: 600;")
        self.view = SyncView()
        lay.addWidget(self.caption)
        lay.addWidget(self.view, 1)


class ComparePanel(QWidget):
    """Avant / après des textures du projet ouvert dans l'interface."""

    def __init__(self, run_task: Callable) -> None:
        super().__init__()
        self._run_task = run_task                 # (fonction, rappel, *arguments) -> tâche de fond
        self.project: Path | None = None
        self.items: list[CompareItem] = []
        self._shown = ""                          # id + chemin de l'image affichée
        self._seq = 0                             # chargement en cours (le dernier demandé gagne)
        self._syncing = False
        self._refreshing = False
        self._pending_follow = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(800)               # regroupe les événements rapprochés
        self.timer.timeout.connect(self._refresh_now)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.setSpacing(6)
        top = QHBoxLayout()
        title = QLabel(tr("AVANT / APRÈS"))
        title.setStyleSheet("font-weight: 800; letter-spacing: 1px;")
        self.prev_btn, self.next_btn = QToolButton(), QToolButton()
        self.prev_btn.setText("◀")
        self.next_btn.setText("▶")
        self.prev_btn.setToolTip(tr("Texture précédente"))
        self.next_btn.setToolTip(tr("Texture suivante"))
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(1))
        self.combo = QComboBox()
        self.combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo.setMinimumContentsLength(18)
        self.combo.currentIndexChanged.connect(self._on_pick)
        self.only_ai = QCheckBox(tr("IA seulement"))
        self.only_ai.setToolTip(tr("Ne lister que les textures passées par l'IA"))
        self.only_ai.toggled.connect(lambda _c: self._fill_combo(keep=True))
        self.follow = QCheckBox(tr("Suivre"))
        self.follow.setChecked(True)
        self.follow.setToolTip(tr("Pendant un lancement, afficher chaque texture dès qu'elle est terminée"))
        top.addWidget(title)
        top.addSpacing(6)
        top.addWidget(self.prev_btn)
        top.addWidget(self.combo, 1)
        top.addWidget(self.next_btn)
        top.addWidget(self.only_ai)
        top.addWidget(self.follow)
        lay.addLayout(top)

        views = QHBoxLayout()
        views.setSpacing(6)
        self.left = ImagePane(tr("Original"))
        self.right = ImagePane(tr("Résultat"))
        views.addWidget(self.left, 1)
        views.addWidget(self.right, 1)
        lay.addLayout(views, 1)
        for a, b in ((self.left.view, self.right.view), (self.right.view, self.left.view)):
            a.moved.connect(lambda a=a, b=b: self._sync(a, b))

        bottom = QHBoxLayout()
        self.info = QLabel("")
        self.info.setTextFormat(Qt.PlainText)          # noms de fichiers du jeu : jamais du HTML
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: #8a93a3;")
        self.zoom_label = QLabel("")
        self.zoom_label.setStyleSheet("color: #8a93a3;")
        self.fit_btn = QPushButton(tr("Ajuster"))
        self.fit_btn.setToolTip(tr("Image entière (double-clic sur une image)"))
        self.fit_btn.clicked.connect(lambda: self.left.view.fit())
        self.one_btn = QPushButton("1:1")
        self.one_btn.setToolTip(tr("Un pixel du résultat = un pixel de l'écran"))
        self.one_btn.clicked.connect(lambda: self.left.view.set_zoom(1.0))
        bottom.addWidget(self.info, 1)
        bottom.addWidget(self.zoom_label)
        bottom.addWidget(self.fit_btn)
        bottom.addWidget(self.one_btn)
        lay.addLayout(bottom)
        self.empty = QLabel(tr("Aucune texture traitée dans ce projet pour l'instant. Lance le pipeline : "
                               "chaque texture apparaîtra ici dès qu'elle est terminée."))
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setStyleSheet("color: #626a78; padding: 30px;")
        lay.addWidget(self.empty, 1)                    # occupe la place des images tant qu'il n'y en a pas
        self._set_empty(True)

    # ------------------------------------------------------------------ projet et liste
    def set_project(self, project: Path | None) -> None:
        if project == self.project:
            return
        self.project = project
        self.items = []
        self._shown = ""
        self._refresh_now()

    def schedule_refresh(self, follow: bool = False) -> None:
        """Après un événement du traitement (texture terminée) : relit la liste un peu plus tard."""
        self._pending_follow = self._pending_follow or follow
        self.timer.start()

    def _refresh_now(self) -> None:
        if self.project is None:
            self.items = []
            self._fill_combo(keep=False)
            return
        if self._refreshing:                      # une lecture est en cours : on repassera après
            self.timer.start()
            return
        self._refreshing = True
        self._run_task(list_items, self._on_items, self.project)

    def _on_items(self, res) -> None:
        self._refreshing = False
        if isinstance(res, dict):                 # erreur de la tâche de fond
            res = []
        follow = self._pending_follow and self.follow.isChecked()
        self._pending_follow = False
        self.items = list(res)
        self._fill_combo(keep=not follow)

    def _visible_items(self) -> list[CompareItem]:
        return [it for it in self.items if it.is_ai] if self.only_ai.isChecked() else self.items

    def _fill_combo(self, keep: bool) -> None:
        current = self.combo.currentData()
        items = self._visible_items()
        self.combo.blockSignals(True)
        self.combo.clear()
        for it in items:
            mark = {"ai": "● ", "esrgan": "● ", "ai_rejected": "◐ "}.get(it.right_kind, "○ ")
            self.combo.addItem(mark + it.name, it.id)
            self.combo.setItemData(self.combo.count() - 1, QColor(KIND_COLORS.get(it.right_kind, "#d9dce3")),
                                   Qt.ForegroundRole)
        idx = self.combo.findData(current) if keep and current else -1
        self.combo.setCurrentIndex(idx if idx >= 0 else (0 if items else -1))
        self.combo.blockSignals(False)
        self._set_empty(not items)
        self._on_pick(self.combo.currentIndex())

    def _step(self, d: int) -> None:
        n = self.combo.count()
        if n:
            self.combo.setCurrentIndex((self.combo.currentIndex() + d) % n)

    def _set_empty(self, empty: bool) -> None:
        self.empty.setVisible(empty)
        for w in (self.left, self.right, self.info, self.zoom_label, self.fit_btn, self.one_btn, self.prev_btn,
                  self.next_btn):
            w.setVisible(not empty)

    # ------------------------------------------------------------------ affichage
    def current_item(self) -> CompareItem | None:
        tid = self.combo.currentData()
        return next((it for it in self.items if it.id == tid), None)

    def _on_pick(self, _index: int) -> None:
        it = self.current_item()
        if it is None:
            return
        key = f"{it.id}|{it.right_path}|{it.updated}"
        if key == self._shown:
            return
        self._shown = key
        self._seq += 1
        seq = self._seq
        self.left.caption.setText(tr("Original — {w}×{h}", w=it.src_size[0], h=it.src_size[1]))
        self.right.caption.setText(tr("{label} — chargement…", label=it.right_label))
        self.right.caption.setStyleSheet(f"font-weight: 600; color: {KIND_COLORS.get(it.right_kind, '#d9dce3')};")
        self.info.setText(it.info)
        self._run_task(self._load_pair, lambda res, seq=seq, it=it: self._on_loaded(res, seq, it), it)

    @staticmethod
    def _load_pair(it: CompareItem) -> dict:
        out = {}
        for key, path in (("left", it.src_path), ("right", it.right_path)):
            try:
                out[key] = load_qimage(path)
            except Exception as exc:              # fichier déplacé, DDS illisible…
                out[key] = None
                out[key + "_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def _on_loaded(self, res: dict, seq: int, it: CompareItem) -> None:
        if seq != self._seq:                      # une autre texture a été choisie entre-temps
            return
        left, right = res.get("left"), res.get("right")
        ref = right if right is not None and not right.isNull() else left
        if ref is None or ref.isNull():
            self.right.caption.setText(tr("{label} — image illisible", label=it.right_label))
            return
        size = (ref.width(), ref.height())
        first = self.left.view.item.pixmap().isNull()
        old_size = (self.left.view.sceneRect().width(), self.left.view.sceneRect().height())
        self.left.view.set_image(QPixmap.fromImage(left) if left is not None else None, size)
        self.right.view.set_image(QPixmap.fromImage(right) if right is not None else None, size)
        self.right.caption.setText(tr("{label} — {w}×{h}", label=it.right_label, w=size[0], h=size[1]))
        if res.get("left_error"):
            self.left.caption.setText(tr("Original illisible : {err}", err=res["left_error"][:120]))
        if first or old_size != size:
            self.left.view.fit()                  # nouvelle taille : image entière ; sinon zoom gardé
        else:
            self._sync(self.left.view, self.right.view)

    def _sync(self, src: SyncView, dst: SyncView) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            if abs(dst.zoom() - src.zoom()) > 1e-9:
                dst.setTransform(src.transform())
                dst._update_filter()
            dst.centerOn(src.center_point())
            self.zoom_label.setText(f"{src.zoom() * 100:.0f} %")
        finally:
            self._syncing = False
