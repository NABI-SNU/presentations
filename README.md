# presentations
Private source code and assets for presentations on SNU CN, the study in SNU Medical School for Computational Neuroscience. 

## Organization

All top-level directories are named as follows: `"{YYYYMMDD}_{Initials of Presentor}_{Initials of First Author}"`. 
Documents (`html`, `pdf`, `doc(x)`, `ppt(x)`, `md`) lies in this directory, while `src` (source code), `assets` (images, videos and audio), and `notebooks` (jupyter notebooks) lie within the corresponding files. 
To represent this as a tree-like schematic, 

```
📁 ProjectRoot/
├── 📁 20250330_AB_CD/
│   ├── 📄 summary.pdf
│   ├── 📄 slides.pptx
│   ├── 📄 notes.md
│   ├── 📁 src/
│   │   └── 🧾 analysis.py
│   ├── 📁 assets/
│   │   ├── 🖼️ diagram.png
│   │   └── 🎞️ concept_video.mp4
│   └── 📁 notebooks/
│       └── 📓 exploration.ipynb
├── 📁 20250212_EF_GH/
│   ├── 📄 report.docx
│   ├── 📁 src/
│   │   └── 🧾 model_training.py
│   ├── 📁 assets/
│   │   └── 🖼️ laplaces_demon_ghibli.png
│   └── 📁 notebooks/
│       └── 📓 simulation.ipynb
```
