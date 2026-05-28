# Devfactor — maintenir notre fork à jour avec l'upstream vLLM

Ce fichier explique **où sont nos modifications**, et **comment rebaser notre travail
sur une version plus récente de vLLM** sans rien perdre. Lis-le en entier avant de
toucher aux branches.

---

## 1. Où sont nos changements ?

Tout notre travail vit sur **UNE seule branche** : `devfactor/custom`.

```
origin/main (vllm-project/vllm)  ← upstream officiel
   │
   └─ d4004455  [Kernel] Remove NormGateLinear (#43554)   ← NOTRE BASE (vLLM 0.21.1)
        │
        ├─ 6cb9f934  WIP: MTP + pipeline parallelism — Qwen3_5MTP (A1 + A3 embed)
        ├─ e8a76168  WIP #18019: MTP spec decode en pipeline parallel (async)
        ├─ dcf008a0  WIP #18019: fix hang PP+MTP (broadcast guard)
        └─ 737b6c1f  WIP #18019: fix bug #6 (num_computed correction)   ← HEAD
```

- **4 commits custom** au-dessus de la base upstream `d4004455` (vLLM **0.21.1**).
- Poussés sur notre fork : `jrabau/vllm`, branche `devfactor/custom`.

### Remotes
| Remote | URL | Rôle |
|--------|-----|------|
| `origin` | `github.com/vllm-project/vllm` | upstream officiel (lecture) |
| `jrabau` | `github.com/jrabau/vllm` | NOTRE fork (push) |

### Les 9 fichiers que nous modifions
```
vllm/model_executor/models/qwen3_5_mtp.py      # MTP+PP : embed/lm_head sur last rank (P4)
vllm/v1/worker/gpu_model_runner.py             # fix #6 : broadcast PP (valid_count col1), positions
vllm/v1/worker/gpu_input_batch.py              # num_accepted optimiste
vllm/v1/spec_decode/llm_base_proposer.py       # share embed/lm_head gardé PP
vllm/v1/core/sched/scheduler.py                # fix livelock load_kv_async (#41515)
vllm/v1/engine/core.py                         # hooks idle-VRAM (guarded ImportError)
vllm/v1/kv_offload/tiering/spec.py             # relax guard block-size hybride (tier disque)
tests/v1/worker/test_gpu_model_runner.py       # tests broadcast/receive PP
tests/v1/kv_offload/test_tiering_offloading.py # tests tiering
```

> ⚠️ Le reste de notre custom KV offload (fs_bounded, CostAwarePolicy, DevfactorScheduler,
> idle_vram) N'EST PAS dans ce fork : il vit dans le **package séparé**
> `devfactor-inference/packages/devfactor_vllm` (installé via `pip install -e`).
> Ce fork ne contient que les **patchs cœur** non extractibles en plugin.

---

## 2. ⚠️ Le piège des `.so` compilés (À LIRE)

Nos changements sont **100 % Python** (aucun `.cu`/`.cpp`). MAIS le repo contient aussi
des **kernels compilés** (`vllm/*.abi3.so` : `_C`, `_moe_C`, flash-attn, cumem...) qui
sont **alignés sur la base `d4004455` (0.21.1)**, compilés le 25 mai.

**Si tu rebases nos `.py` sur une base upstream plus récente, les `.so` ne correspondent
plus** → crash possible au runtime (les bindings C++ `torch.ops._C.*` ont pu changer).

Pour vérifier si un rebase upstream touche du C++ (donc impose une recompilation) :
```bash
git diff --name-only d4004455..<nouvelle-base> | grep -iE '\.(cu|cuh|cpp|cc|h|hpp)$|csrc/|CMakeLists'
```
- **Aucune sortie** → recompilation inutile, les `.so` actuels suffisent.
- **Des fichiers listés** → il FAUT recompiler (voir §5).

C'est exactement pourquoi on est resté sur `d4004455` lors de la session du 27/05 :
les 62 commits upstream suivants touchaient 14 fichiers C++.

---

## 3. Procédure : mettre à jour depuis l'upstream

But : passer nos 4 commits d'une vieille base upstream à une base plus récente.

### Étape 0 — Filet de sécurité (TOUJOURS)
```bash
cd /home/jonathan/Documents/sources/vllm
git checkout devfactor/custom
git branch backup/avant-rebase-$(date +%Y%m%d)        # branche de secours
git format-patch d4004455..HEAD -o /tmp/devfactor-patches  # patchs sur disque
```

### Étape 1 — Récupérer l'upstream
```bash
git fetch origin                  # met à jour origin/main
git log --oneline origin/main -5  # repérer le commit cible (ou un tag : git fetch origin --tags)
```
Choisis le commit/tag de base voulu, appelons-le `<NEW_BASE>` (ex `origin/main` ou `vX.Y.Z`).

### Étape 2 — Rebaser nos 4 commits sur la nouvelle base
```bash
git rebase --onto <NEW_BASE> d4004455 devfactor/custom
```
Lecture : « rejoue les commits qui sont après `d4004455` (= nos 4) par-dessus `<NEW_BASE>` ».

### Étape 3 — Gérer les conflits (voir §4 en détail)
Si le rebase s'arrête sur un conflit, résous, puis :
```bash
git add <fichiers-résolus>
git rebase --continue
```
Pour tout annuler et revenir à l'état d'avant :
```bash
git rebase --abort
```

### Étape 4 — Vérifier que notre travail est intact
```bash
# nos changements clés doivent être présents :
grep -n "self.lm_head = PPMissingLayer" vllm/model_executor/models/qwen3_5_mtp.py
grep -n "valid_sampled_token_count_gpu = bundle" vllm/v1/worker/gpu_model_runner.py
```

### Étape 5 — Recompiler les `.so` SI nécessaire (cf. §2 et §5)

### Étape 6 — Publier
```bash
git push --force-with-lease jrabau devfactor/custom   # --force car l'historique a changé
```
> `--force-with-lease` (et pas `--force` nu) : refuse de pousser si quelqu'un d'autre a
> poussé entre-temps. Plus sûr.

---

## 4. En cas de conflit

Un conflit = upstream a modifié les mêmes lignes que nous. Git marque le fichier ainsi :
```
<<<<<<< HEAD                 (la version upstream <NEW_BASE>)
   code upstream
=======
   notre code
>>>>>>> <hash> (notre commit)
```

**Comment décider :**
1. Ouvre le fichier, lis les deux versions.
2. La plupart de nos conflits sont sur `qwen3_5_mtp.py` (embed/lm_head) — **notre intention
   prime** : `self.lm_head = PPMissingLayer()` (jamais allouer la tête MTP, elle est
   partagée du target). Garde notre bloc + le commentaire `devfactor P4`.
3. Si upstream a refactoré autour, **fusionne manuellement** : garde notre logique mais
   adapte-la aux noms/signatures upstream récents.
4. Retire les marqueurs `<<<<<<<`, `=======`, `>>>>>>>`.
5. `git add <fichier>` puis `git rebase --continue`.

**Vérifier qu'un commit n'a pas été dénaturé** (utile après résolution) — comparer le
contenu via patch-id, indépendant de la base :
```bash
git log <ancienne-base>..backup/avant-rebase-AAAAMMJJ -p | git patch-id --stable
git log <NEW_BASE>..devfactor/custom        -p | git patch-id --stable
```
Les commits NON conflictuels doivent avoir le **même patch-id** des deux côtés. Seuls
ceux qu'on a résolus diffèrent (normal).

**Astuce — réduire les conflits récurrents :** activer `rerere` (git mémorise tes
résolutions et les rejoue automatiquement au prochain rebase) :
```bash
git config --global rerere.enabled true
```

---

## 5. Recompiler les `.so` (si §2 l'impose)

Nos changements sont Python, mais si la **nouvelle base upstream** a modifié des kernels,
il faut reconstruire les extensions. Dans le conteneur `vllm-build` (qui a CUDA + deps) :

```bash
# le fork est monté en bind sur /workspace/vllm
docker exec vllm-build bash -lc '
  cd /workspace/vllm &&
  MAX_JOBS=24 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST=12.0 \
  pip install -e . --no-build-isolation
'
```
- `MAX_JOBS=24 NVCC_THREADS=1` → ninja `-j 24` sur nos 32 cœurs.
  ⚠️ PIÈGE `setup.py` : `num_jobs = MAX_JOBS // NVCC_THREADS`. Avec `NVCC_THREADS=8` on
  retombait à `-j 3` (CPU sous-utilisé). Mettre `NVCC_THREADS=1`.
- `TORCH_CUDA_ARCH_LIST=12.0` → compile UNIQUEMENT pour nos GPU (RTX 5060Ti/5070Ti =
  sm_120), pas les 9 archis par défaut. Gros gain de temps.

Le `pip install -e` régénère les `.so` et le fichier `__editable__.vllm-...+precompiled.pth`
qui fait pointer l'import vers `/workspace/vllm`.

---

## 6. Rappels environnement

- **`vllm-build`** = image `vllm/vllm-openai:cu130-nightly` (socle CUDA) + ce fork monté en
  bind (`/workspace/vllm`) + le package `devfactor_vllm` copié dans `/opt`. C'est là qu'on teste.
- **La prod** (`devfactor-inference/mgmt/catalogue.yaml`, `qwen36-vllm`) tourne sur l'image
  NUE par défaut. Pour lui donner notre custom : décommenter le bloc `volumes:` documenté
  dans le catalogue (monte ce fork + installe le package).
- Le fork prime sur le vLLM de l'image via `pip install -e` (crée le `.pth`), PAS via
  `PYTHONPATH`.

---

## 7. Aide-mémoire express

| Je veux… | Commande |
|----------|----------|
| Voir nos commits | `git log --oneline d4004455..devfactor/custom` |
| Voir nos fichiers modifiés | `git diff --name-only d4004455..devfactor/custom` |
| Sécuriser avant manip | `git branch backup/X && git format-patch d4004455..HEAD -o /tmp/p` |
| Récupérer upstream | `git fetch origin` |
| Rebaser sur du neuf | `git rebase --onto <NEW_BASE> d4004455 devfactor/custom` |
| Annuler un rebase | `git rebase --abort` |
| Upstream touche-t-il du C++ ? | `git diff --name-only d4004455..<NEW_BASE> \| grep -E '\.(cu\|cpp\|h)$\|csrc/'` |
| Publier après rebase | `git push --force-with-lease jrabau devfactor/custom` |
