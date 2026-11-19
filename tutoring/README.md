# Tutoring sessions

Live-coding notebooks for the Representation Learning tutoring sessions. You code along on your own laptop. Each session has its own folder, `tutoring/sessionNN/`.

## Session 1: OOP → PyTorch Lightning → TensorBoard (`session01/`)

You've already run `class NN(pl.LightningModule)` in the lectures. Today you'll understand every line of it: the Python classes underneath it, why Lightning asks you to write methods instead of a training loop, and how to watch and compare your runs in TensorBoard.

| Notebook | Topic |
|---|---|
| `session01/00_setup_check.ipynb` | Is everything installed? |
| `session01/01_oop.ipynb` | Classes, `self`, inheritance, `super().__init__()`, hooks, `__call__`, `__getitem__` |
| `session01/02_bridge_to_pytorch.ipynb` | `nn.Module` is the same OOP, plus a plain training loop and all its plumbing |
| `session01/03_lightning.ipynb` | Your own mini `Trainer`, then `LightningModule` + `pl.Trainer`, and the hook order |
| `session01/04_tensorboard.ipynb` | `self.log`, `TensorBoardLogger`, comparing runs |

Everything trains on a laptop CPU in seconds, using `data/mnist_small.npz` (6,000 training and 1,000 test MNIST digits).

**Can't install anything, or your setup broke?** Click the "Open in Colab" badge at the top of any notebook. It runs in your browser, so you can keep following along.

---

## 1. Before the session: install the tools (≈ 20 minutes)

Already have Git, Miniforge (or Anaconda) and VS Code with the Python and Jupyter extensions? Skip to [section 2](#2-before-the-session-update-the-repo-and-create-the-environment).

### Windows

1. **Git**: download from <https://git-scm.com/download/win> and run the installer. Keep the default answer on every screen, except one: on the *"Choosing the default editor"* screen, pick **Visual Studio Code**.
2. **Miniforge**: download `Miniforge3-Windows-x86_64.exe` from <https://github.com/conda-forge/miniforge/releases/latest> and run it. Choose **"Just Me"** and keep the defaults.
3. **VS Code**: install it from <https://code.visualstudio.com>. Open it, go to the Extensions tab on the left, and install **Python** and **Jupyter** (both are published by Microsoft).

👉 On Windows, run **every command** in this README in the **Miniforge Prompt**. Find it in the Start menu by typing "Miniforge". Don't use the regular Command Prompt or PowerShell, because `conda` won't work there.

### Mac

1. **Git**: open **Terminal** and run `git --version`. If git isn't installed yet, macOS will offer to install it, so click *Install*.
2. **Miniforge**: in Terminal, run:
   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
   Press Enter or type `yes` at each question. When the installer finishes, **close Terminal and open it again**.
3. **VS Code**: install it from <https://code.visualstudio.com>, then install the **Python** and **Jupyter** extensions from its Extensions tab.

### Check that it worked (Windows and Mac)

```bash
git --version
conda --version
```

If both commands print a version number, you're done ✅.

---

## 2. Before the session: update the repo and create the environment

In the Miniforge Prompt (Windows) or Terminal (Mac), go into your copy of the course repo and run:

```bash
cd representation-learning-course
git pull --no-edit
conda env create -f tutoring/environment.yml
```

Don't have the repo yet? Run `git clone https://github.com/SivanSchwartz/representation-learning-course.git` first, then the commands above.

The last command downloads Python, PyTorch (the CPU version), Lightning and TensorBoard into an environment called `replearn`. **PyTorch is a large download, so do this before the session, not during it.** You only need to do it once.

Then open the repo in VS Code: **File → Open Folder…** and select the `representation-learning-course` folder. Open `tutoring/session01/00_setup_check.ipynb`, click **Select Kernel** (top right), then **Python Environments**, then **replearn**. Run the cells. You should see `You're ready! 🎉`.

---

## 3. During the session

- **Fell behind?** Run `git pull --no-edit`, then open the matching notebook in `tutoring/session01/solutions/`. Solutions are released one at a time during the session. They always arrive as **new files**, so pulling never overwrites your own work.
- **Changed a lecture notebook and `git pull` complains?** Make a copy of your edited notebook under a new name (e.g. `my_05_moco.ipynb`), then run `git checkout -- <the original file>` and pull again.

## 4. Cheat sheet

| Command | What it does |
|---|---|
| `git status` | Shows what has changed since your last commit. Run it whenever you're unsure. |
| `git pull --no-edit` | Downloads updates from the course repo (new notebooks, solutions) |
| `conda activate replearn` | Switch to the tutoring environment (needed before running `tensorboard` in a terminal) |
| `conda env list` | List your environments. The active one has a `*`. |
| `conda env update -f tutoring/environment.yml` | Install anything that was added to `environment.yml` |
| `tensorboard --logdir tutoring/session01/tb_logs` | Start TensorBoard, then open <http://localhost:6006> in your browser |
