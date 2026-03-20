#!/bin/bash
#SBATCH --job-name=parallel_rm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # Prova con 16 core; aumenta se il FS è molto veloce
#SBATCH --mem=4G                    # Parallel consuma un po' di RAM per gestire le liste
#SBATCH --time=03:00:00            # I file piccoli sono lenti, dai abbastanza tempo
#SBATCH --partition=shortq           # Usa la tua partizione di default
#SBATCH --output=parallel_rm.out

# 1. Definisci le cartelle da eliminare
#TARGETS=("/mnt/beegfs01/scratch/a_morelli/datasets/pdfs/Q13" "/mnt/beegfs01/scratch/a_morelli/parallel_censoring/*")censored_images  ref_pdf_backup  run_data_q13

# 1. Definisci la base comune e le sottocartelle
BASE_DIR="/mnt/beegfs01/scratch/a_morelli/parallel_censoring"

# Definiamo solo i nomi delle cartelle relative alla BASE_DIR
TARGETS=(
    "$BASE_DIR/ref_pdf_backup" #i cancel these before so that i can proceed with the next extraction in parallel
    "$BASE_DIR/run_data_q13"
    "$BASE_DIR/censored_images/13"
    "$BASE_DIR/censored_images/archived_13"
)

module load parallel/20250222

# 3. Esecuzione parallela
# Spiegazione:
# 'find' elenca il contenuto delle cartelle
# '-nd' (non-deterministic/no-directories) o limitando la profondità
# 'parallel' lancia rm -rf su più core contemporaneamente

for dir in "${TARGETS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Inizio eliminazione di: $dir"
        
        # Usiamo find per passare i file a parallel
        # -j $SLURM_CPUS_PER_TASK usa esattamente i core allocati da Slurm
        find "$dir" -mindepth 1 -maxdepth 1 | parallel -j $SLURM_CPUS_PER_TASK rm -rf
        
        # Elimina la cartella radice rimasta vuota
        rm -rf "$dir"
        echo "Completato: $dir"
    else
        echo "Salto $dir: cartella non trovata."
    fi
done