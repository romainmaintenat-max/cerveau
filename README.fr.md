# Cerveau

**Une mémoire à long terme locale et souveraine pour agents IA.**
Indexez vos notes, laissez vos agents retenir ce qui compte, et cherchez tout par le sens —
sur votre propre machine, sans service cloud et sans facture d'API.

Cerveau se branche sur n'importe quel client MCP (Claude Desktop, Claude Code, etc.) et lui donne :

- `search_memory` — recherche par le sens dans vos documents et dans les souvenirs de l'agent ;
- `remember` — retenir un fait, une décision ou une leçon durable ;
- `brain_status` — ce qu'il contient.

*[English version: README.md](README.md)*

## Pourquoi un outil de mémoire de plus ?

Cerveau reprend les choix d'un système de mémoire personnel qui tourne depuis des mois sur un serveur IA domestique.
Ce paquet en est une réécriture propre et autonome ; chaque principe vient de quelque chose qui a mal tourné :

- **Le journal fait foi.** Chaque souvenir est écrit dans un journal en ajout seul, synchronisé sur disque,
  *avant* d'arriver dans la base vectorielle. Qdrant perdu, modèle d'embedding changé : `cerveau replay`
  reconstruit tous les souvenirs (les documents se reconstruisent avec `cerveau index`, depuis vos fichiers).
  Le journal garde le texte, jamais les vecteurs. Chaque ligne porte un numéro de séquence et une somme de
  contrôle : une perte au milieu du journal est détectée, et les dégâts qu'un crash laisse en fin de fichier
  (ligne à moitié écrite, octets nuls) sont reconnus, mis de côté et jamais collés à l'écriture suivante.
- **Collections scellées.** Le modèle et la dimension sont inscrits à la création. Repartir avec un autre
  modèle, ou sur des collections que Cerveau n'a pas créées, est refusé avant toute écriture.
- **Aucune troncature silencieuse.** Un texte trop long pour le modèle est refusé avec un message clair.
- **L'agent ajoute, l'humain retire.** Par MCP, un agent cherche et retient. Archiver, restaurer, indexer et
  reconstruire se font uniquement en ligne de commande. L'archivage est réversible ; seul `cerveau purge`, lancé par vous, efface.
- **Sûr par défaut.** Qdrant exige une clé API et n'écoute que sur 127.0.0.1. Voir [SECURITY.md](SECURITY.md).

## Prérequis

| Quoi | Version | Remarques |
|---|---|---|
| Python | 3.11 ou plus | le `python3` livré avec macOS est plus ancien, comme ceux d'Ubuntu 22.04 (3.10) et de Debian 11 (3.9) : installez-en un récent. Debian/Ubuntu : `sudo apt install python3-venv` |
| Docker + Docker Compose v2 | récent | fait tourner Qdrant (`docker compose`, pas `docker-compose`) |
| [Ollama](https://ollama.com) | récent | fait tourner le modèle d'embedding ; il doit être lancé |
| Modèle d'embedding | `bge-m3` (défaut) : multilingue, plus de 100 langues, licence MIT<br>`snowflake-arctic-embed2` (alternative) : même taille et même dimension ; changer de modèle demande de réindexer | tourne sur CPU ; un GPU accélère seulement l'indexation |

Pour utiliser l'alternative : `ollama pull snowflake-arctic-embed2` puis `CERVEAU_EMBED_MODEL=snowflake-arctic-embed2` dans `.env` (`CERVEAU_VECTOR_SIZE` reste à `1024`).

## Démarrage rapide

```bash
git clone https://github.com/romainmaintenat-max/cerveau.git
cd cerveau
python3 -m venv .venv && source .venv/bin/activate    # Windows : voir plus bas
pip install -e .

cp .env.example .env && chmod 600 .env                # Windows : copy .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # à coller dans .env, CERVEAU_QDRANT_API_KEY

docker compose up -d                                  # laissez à Qdrant quelques secondes pour démarrer
ollama pull bge-m3

cerveau init
cerveau index examples/notes
cerveau search "à quelle température cuire le pain ?"
```

Sous Windows (PowerShell) : `py -m venv .venv`, puis `.venv\Scripts\Activate.ps1` (si les scripts sont
bloqués : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`), et `copy .env.example .env`. À la place de `chmod 600` :
`icacls .env /inheritance:r /grant:r "${env:USERNAME}:F"` (retire les droits hérités et donne l'accès complet à vous seul).

Le premier résultat doit venir de `bread.md` et parler de 240 °C (la note est en anglais : la recherche
passe d'une langue à l'autre). Pour arrêter Qdrant : `docker compose down` (jamais `down -v`, qui efface
le volume Qdrant ; le journal survit, mais il faudra `init`, `replay` et réindexer). Indexez ensuite vos propres dossiers. Lancez `cerveau` depuis le dossier
qui contient `.env`, ou définissez `CERVEAU_ENV_FILE` (voir Configuration). Dans un nouveau terminal,
réactivez l'environnement virtuel (`source .venv/bin/activate`, ou `.venv\Scripts\Activate.ps1` sous
Windows), ou appelez `.venv/bin/cerveau` par son chemin complet.
`.env` contient la clé Qdrant : gardez-le lisible par vous seul (`chmod 600`, ou `icacls` sous Windows). Si vous choisissez une clé vous-même, évitez le `$` :
Docker Compose la lirait autrement que Cerveau. Gardez `.env` en UTF-8 : Docker Compose ne lit pas l'UTF-16.

## Ligne de commande

| Commande | Rôle |
|---|---|
| `cerveau init` | crée les collections, vérifie la configuration |
| `cerveau index <dossier ou fichier> [--prune]` | indexe les `.md`, `.markdown`, `.txt` ; saute les fichiers inchangés ; les fichiers supprimés depuis restent trouvables jusqu'à `--prune` (un avertissement le dit) ; refuse un disque entier, le dossier personnel, le dossier de tous les utilisateurs et les dossiers système (`/etc`, `/usr`, `C:\Windows`…) ; code de sortie 1 si un fichier n'a pas pu être indexé (listé dans `errors`) ; `--prune` (dossiers seulement) retire ce qui n'y est plus indexable : fichiers supprimés, et fichiers désormais ignorés (cachés, exclus) ; rien n'est retiré si un sous-dossier n'a pas pu être lu, ni si le dossier ne contient plus aucun fichier texte (un disque non monté paraît vide : `unindex` si les fichiers ont vraiment disparu) |
| `cerveau unindex <chemin>` | retire ce qui a été indexé sous un chemin, même s'il n'existe plus (dossier renommé ou déplacé) |
| `cerveau remember "<texte>" [--kind decision]` | retient un souvenir |
| `cerveau search "<question>" [-k 5 (1-50 ; 1-20 par MCP)] [--scope all\|docs\|memories] [--include-archived] [--json]` | cherche par le sens |
| `cerveau list [--actor mcp] [--include-archived] [--json]` | tous les souvenirs du journal, en entier, avec leur auteur (`mcp` = un client IA, `cli` = vous, `adopted` = récupéré par `adopt-orphans`, affiché avec l'auteur inscrit dans Qdrant) |
| `cerveau list --orphans` | les souvenirs présents dans Qdrant mais absents de ce journal |
| `cerveau archive <id>` / `cerveau restore <id>` | masque / réaffiche un souvenir (réversible ; le texte reste dans le journal) |
| `cerveau purge <id> --confirm` | retire un souvenir du cerveau actif (journal et Qdrant), pour un secret enregistré par erreur : réécrit le journal sans lui et le retire de Qdrant, dans tous les préfixes Cerveau de ce Qdrant. Les sauvegardes antérieures, et les fichiers internes de Qdrant tant qu'il ne les a pas réécrits, contiennent encore le texte : voir [SECURITY.md](SECURITY.md) pour les effacer aussi |
| `cerveau replay` | réécrit dans Qdrant tous les souvenirs du journal |
| `cerveau status` | comptes et configuration ; prévient quand Qdrant est en retard sur le journal |
| `cerveau adopt-orphans` | recopie dans le journal les souvenirs présents seulement dans Qdrant (voir `list --orphans`) ; leur auteur devient `adopted` |
| `cerveau reset-collections --confirm <préfixe> [--discard-orphans]` | supprime les collections Qdrant de ce préfixe (journal et fichiers conservés), pour repartir après une mauvaise configuration. Refusé tant que Qdrant contient des souvenirs absents du journal : `adopt-orphans` d'abord, ou `--discard-orphans` pour les perdre |
| `cerveau --version` | |

Un texte qui commence par `-` se place après `--` : `cerveau remember -- "-5 °C cette nuit"`.

Les fichiers doivent être en UTF-8 (ou UTF-16 avec BOM, l'« Unicode » du Bloc-notes) ; les autres encodages, et les fichiers binaires, sont signalés, pas devinés.
Sous le dossier donné, les fichiers et dossiers cachés (`.git`, `.brouillon.md`…), `node_modules`, `venv`,
`site-packages`, `__pycache__` et les liens symboliques sont ignorés (un dossier que vous nommez vous-même
est parcouru même s'il est caché ; un *fichier* caché nommé directement est refusé). Les fichiers de plus
de 20 Mo sont ignorés et signalés. Un dossier illisible est signalé, les autres sont indexés.
Un document est identifié par son chemin absolu : indexer un dossier puis un de ses sous-dossiers ne crée
pas de doublons. `index --prune` et `unindex` retirent des morceaux de *documents*, jamais des souvenirs.

## Brancher un client IA

Le serveur MCP est lancé par le client (stdio), depuis un dossier que vous ne maîtrisez pas. Il ne lit donc
**jamais** de `.env` dans son dossier courant : donnez-lui `CERVEAU_ENV_FILE`, le chemin absolu de votre `.env`.
Lancez `cerveau init` une fois avant de brancher un client : par MCP, rien n'est jamais créé (pas même par `remember`). Les erreurs de configuration s'affichent sur la
sortie d'erreur du serveur (la plupart des clients la montrent dans leurs journaux MCP).

**Claude Desktop** — dans `claude_desktop_config.json` (menu Réglages → Développeur → Modifier la config ;
sous macOS `~/Library/Application Support/Claude/`, sous Windows `%APPDATA%\Claude\`) :

```json
{
  "mcpServers": {
    "cerveau": {
      "command": "/chemin/absolu/vers/cerveau/.venv/bin/cerveau-mcp",
      "env": { "CERVEAU_ENV_FILE": "/chemin/absolu/vers/cerveau/.env" }
    }
  }
}
```

Sous Windows : `C:\\chemin\\vers\\cerveau\\.venv\\Scripts\\cerveau-mcp.exe` et `C:\\chemin\\vers\\cerveau\\.env`
(en JSON, chaque antislash s'écrit deux fois).

**Claude Code** :

```bash
claude mcp add -s user cerveau -e CERVEAU_ENV_FILE=/chemin/absolu/vers/cerveau/.env -- /chemin/absolu/vers/cerveau/.venv/bin/cerveau-mcp
```

Sous Windows : `claude mcp add -s user cerveau -e CERVEAU_ENV_FILE=C:\chemin\vers\cerveau\.env -- C:\chemin\vers\cerveau\.venv\Scripts\cerveau-mcp.exe`

## Configuration

Les réglages vivent dans `.env` (voir [.env.example](.env.example)). Seules les lignes `CERVEAU_*` sont lues,
et un nom `CERVEAU_*` proche d'un vrai réglage est refusé : une faute de frappe ne retombe jamais en silence sur une valeur par défaut.
Le fichier lu est `CERVEAU_ENV_FILE` s'il est défini (il doit exister) ; sinon la ligne de commande — pas le
serveur MCP — lit le `.env` du dossier courant. Une variable d'environnement non vide l'emporte sur le fichier. Tout
nom qui commence par `CERVEAU` est vérifié : à 3 modifications ou moins d'un réglage (faute de frappe, mauvais
séparateur), fait d'une partie des mots d'un réglage ou de plus (`CERVEAU_ALLOW_WRITE`, `CERVEAU_DATA`), ou parlant
d'écriture ou de lecture seule (`CERVEAU_READONLY`), c'est une erreur ; tout autre nom (`CERVEAU_TOKEN`) appartient
à un autre logiciel et il est ignoré avec un avertissement.

| Variable | Défaut | |
|---|---|---|
| `CERVEAU_ENV_FILE` | — | chemin absolu du `.env` à charger (variable d'environnement uniquement) |
| `CERVEAU_QDRANT_URL` | `http://127.0.0.1:6333` | |
| `CERVEAU_QDRANT_API_KEY` | — | obligatoire pour un Qdrant non local |
| `CERVEAU_OLLAMA_URL` | `http://127.0.0.1:11434` | une adresse non locale affiche un avertissement : tout le texte y part |
| `CERVEAU_OLLAMA_TIMEOUT` | `300` | secondes par requête d'embedding ; à augmenter sur un CPU lent |
| `CERVEAU_EMBEDDER` | `ollama` | `hash` est un faux embedder, non sémantique, réservé aux tests |
| `CERVEAU_EMBED_MODEL` / `CERVEAU_VECTOR_SIZE` | `bge-m3` / `1024` | doivent correspondre |
| `CERVEAU_MAX_TEXT_CHARS` / `CERVEAU_CHUNK_MAX_CHARS` | `3000` / `1500` | un morceau doit laisser 256 caractères au titre. Un texte dense (code, tableaux) peut demander moins |
| `CERVEAU_PREFIX` | `cerveau` | collections `<prefix>_docs`, `_memories`, `_meta`. Ne sépare **pas** les journaux : voir plus bas |
| `CERVEAU_DATA_DIR` | `~/.cerveau` | le journal y vit — **sauvegardez-le**. les variables sont développées à la manière du système (`$VAR` sous Linux/macOS, `%VAR%` sous Windows) ; un chemin relatif dans `.env` part du dossier du `.env` |
| `CERVEAU_MCP_ALLOW_WRITE` | `true` | `false` : pas d'outil `remember`, et le serveur MCP n'écrit plus rien du tout (hors son fichier de log) |
| `CERVEAU_LOG` | `on` | `off` : pas de fichier de log (voir [Signaler un bug](#signaler-un-bug)) |
| `CERVEAU_QDRANT_PORT` | `6333` | port de Qdrant sur cette machine, lu seulement par `docker-compose.yml` (changer `CERVEAU_QDRANT_URL` avec) |

Le serveur MCP lit ses réglages une seule fois, quand le client IA le lance : **après avoir modifié le
`.env`, redémarrez le client IA** (Claude Desktop, Claude Code…). D'ici là, les anciens réglages
restent en vigueur, `CERVEAU_MCP_ALLOW_WRITE=false` compris.


**Un journal par dossier de données, pas par préfixe.** Les souvenirs appartiennent à `CERVEAU_DATA_DIR` ;
un préfixe n'est qu'un jeu de collections Qdrant construit à partir d'eux. Tous les préfixes qui partagent
le même dossier de données voient les mêmes souvenirs après `cerveau replay` — c'est ainsi que fonctionne
un changement de modèle. Pour un cerveau séparé (essais, autre personne), prenez un autre `CERVEAU_DATA_DIR`.

**Changer de modèle d'embedding** : prenez un nouveau `CERVEAU_PREFIX`, relancez `cerveau index` sur vos
dossiers et `cerveau replay` pour les souvenirs. Les anciennes collections restent intactes jusqu'à ce que
vous les retiriez (`cerveau reset-collections --confirm <ancien préfixe>` avec les anciens réglages).

**Une première configuration ratée** (modèle non téléchargé, mauvais `CERVEAU_VECTOR_SIZE`) : `cerveau init`
essaie le modèle avant de créer quoi que ce soit, donc rien n'est scellé. Si un préfixe a quand même été
scellé avec de mauvaises valeurs, `cerveau reset-collections --confirm <préfixe>` le retire.

## Quand quelque chose ne va pas

- **`remember` répond que le souvenir « IS saved in the journal, but Qdrant failed »** : ne réessayez pas
  (cela créerait un doublon). Quand Qdrant est revenu, lancez `cerveau replay`.
- **`error: … 401`** : la clé Qdrant est fausse, ou le bon `.env` n'a pas été chargé.
- **`status` affiche `journal_torn_lines`** : un crash a interrompu une écriture à un moment. Les octets
  inachevés restent dans le fichier et sont ignorés. Un crash ne peut pas faire perdre un souvenir que
  `remember` avait confirmé.
  (Une erreur disque qui abîme la toute fin de la dernière ligne peut ressembler à ce cas : gardez des sauvegardes.)
- **« unreadable record (not an interrupted write) »** : le journal a été modifié par autre chose que
  Cerveau (édition, ré-encodage, erreur disque). Cerveau s'arrête au lieu de deviner : restaurez le journal
  depuis votre sauvegarde.
- **Le journal est illisible et vous n'avez pas de sauvegarde** : si Qdrant contient encore les souvenirs,
  reconstruisez le journal à partir de lui : renommez le fichier abîmé (gardez-le le temps d'être sûr ; il contient encore tous les textes, y compris
  ceux purgés depuis), puis lancez
  `cerveau adopt-orphans` — chaque souvenir trouvé dans Qdrant est recopié dans un nouveau journal, avec sa date d'origine.
- **Ce qui n'est *pas* détecté** : un journal raccourci par la fin (sauvegarde plus ancienne restaurée,
  copie interrompue au milieu de sa dernière ligne) ressemble à un journal qui a simplement moins
  d'enregistrements. `status` signale alors que Qdrant contient des souvenirs absents du journal
  (`cerveau list --orphans`).
- **« the brain is busy »** : une autre commande tient un verrou (attente de 60 s). Documents et souvenirs
  ont des verrous séparés : un long `cerveau index` bloque un autre `index`, jamais le `remember` de
  l'agent ; un `replay` bloque `remember`. Les recherches ne sont jamais bloquées.
- **« did not answer within 300 s »** : Ollama est trop lent pour une requête (CPU seul). Augmentez
  `CERVEAU_OLLAMA_TIMEOUT`.
- **L'indexation d'un fichier échoue avec « input length exceeds the context length »** : un texte très
  dense (code, tableaux, hexadécimal, chinois/japonais/coréen) consomme plus de tokens par caractère.
  Baissez `CERVEAU_CHUNK_MAX_CHARS`. La même limite peut refuser un `remember` très dense : rien n'est
  alors écrit, découpez le texte.

`search` renvoie toujours les `k` résultats les plus proches, même quand rien n'est vraiment proche : regardez les scores.

## Signaler un bug

Cerveau écrit un fichier de log (distinct du journal des souvenirs) sur votre machine, dans
`<CERVEAU_DATA_DIR>/logs/` : `cli.log` pour la ligne de commande, `mcp.log` pour le serveur MCP (5 Mo chacun,
puis 3 fichiers plus anciens). Il ne quitte jamais votre ordinateur de lui-même : rien n'est envoyé nulle
part. Il contient les versions, les noms des commandes et des outils, les durées, des nombres, les
avertissements et les erreurs avec l'endroit du code où elles se sont produites — jamais un souvenir, une
recherche, un document ni la clé Qdrant ; votre dossier de données, celui du fichier de réglages et votre
dossier personnel y sont écrits `<data_dir>`, `<env_dir>` et `~`. Pour signaler un bug, lisez le fichier,
puis joignez-le à une
[issue GitHub](https://github.com/romainmaintenat-max/cerveau/issues) avec la commande lancée. Pour couper
le log : `CERVEAU_LOG=off` dans `.env`.

## Fonctionnement

```
vos notes ──► cerveau index ──► morceaux ──► Ollama (embedding) ──► Qdrant  <prefix>_docs
agent ──► remember ──► Ollama (embedding) ──► journal.jsonl (fsync) ──► Qdrant  <prefix>_memories
agent ──► search_memory ──► Ollama ──► les deux collections, fusionnées par score cosinus
```

## Développement

```bash
pip install -e ".[dev]"
pytest                    # tests unitaires : Qdrant en mémoire, faux embedder, pas de GPU
pytest -m integration     # sur un vrai Qdrant : CERVEAU_IT_QDRANT_URL et CERVEAU_IT_QDRANT_API_KEY
                          # et un vrai Ollama (avant une version) : CERVEAU_IT_OLLAMA_URL
ruff check . && ruff format --check .
```

## Limites de cette version

- Fichiers texte seulement (`.md`, `.markdown`, `.txt`). Pas de PDF ni de fichiers Office. Les images
  collées (URI `data:`) et les suites d'au moins 1000 caractères base64 ou hexadécimaux sont remplacées par
  `[binary data omitted]` avant l'indexation (le texte de toute langue est gardé, y compris le chinois ou le
  japonais écrits sans espaces).
- Un seul écrivain à la fois par dossier de données (un fichier verrou l'impose). Gardez `CERVEAU_DATA_DIR`
  sur un disque local : partages réseau et dossiers synchronisés (Dropbox, OneDrive…) ne rendent pas les
  verrous et les ajouts fiables.
- Aucune limite au nombre de souvenirs qu'un agent peut ajouter : relisez-les avec `cerveau list --actor mcp`.
- Pas de détection de doublons ni de nettoyage automatique : vous triez avec `archive` (ou `purge`).
- Le sceau compare le *nom* du modèle (`bge-m3:latest` et `bge-m3` désignent le
  même modèle pour Ollama et sont traités comme un seul). Un modèle retéléchargé avec d'autres poids sous le même nom n'est pas détecté.

## Licence

MIT — voir [LICENSE](LICENSE).
