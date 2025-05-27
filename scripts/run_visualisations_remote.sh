#!/bin/bash

# Użycie: ./run_visualisations_remote.sh input.csv 2 ares plgjanow01

LOCAL_INPUT_FILE="$1"
N_COMPONENTS="$2"
CLUSTER="$3"
LOGIN="$4"

if [ $# -ne 4 ]; then
  echo "Użycie: ./run_visualisations_remote.sh <plik_csv> <n_components> <ares|athena> <login>"
  exit 1
fi

REMOTE_HOST="${LOGIN}@${CLUSTER}.cyfronet.pl"
REMOTE_DIR="/net/people/plgrid/${LOGIN}/visualisations"
REMOTE_INPUT="input.csv"
REMOTE_OUTPUT="output.csv"

echo "[1] Tworzenie folderu na klastrze..."
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

echo "[2] Przesyłanie plików..."
scp "$LOCAL_INPUT_FILE" "$REMOTE_HOST:$REMOTE_DIR/$REMOTE_INPUT"
scp visualisations_script.py "$REMOTE_HOST:$REMOTE_DIR/"
scp slurm_job.sh "$REMOTE_HOST:$REMOTE_DIR/"

echo "[3] Tworzenie job.sh ze zmienioną liczbą komponentów..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && sed 's/{{n_components}}/'$N_COMPONENTS'/g' slurm_job.sh > job.sh"

echo "[4] Wysyłanie zadania do kolejki..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && sbatch job.sh"

echo "[5] Oczekiwanie na zakończenie zadania i pojawienie się pliku wynikowego..."

MAX_WAIT=600
WAIT_INTERVAL=10
ELAPSED=0

while true; do
  FILE_EXISTS=$(ssh "$REMOTE_HOST" "[ -f $REMOTE_DIR/$REMOTE_OUTPUT ] && echo yes || echo no")
  if [ "$FILE_EXISTS" == "yes" ]; then
    echo "[6] Plik wynikowy znaleziony, pobieranie..."
    scp "$REMOTE_HOST:$REMOTE_DIR/$REMOTE_OUTPUT" "./output_pca.csv"
    echo "Plik output_pca.csv został zapisany lokalnie."
    break
  fi

  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "Przekroczono maksymalny czas oczekiwania ($MAX_WAIT sekund)."
    exit 1
  fi

  sleep "$WAIT_INTERVAL"
  ELAPSED=$((ELAPSED + WAIT_INTERVAL))
  echo "Oczekiwanie na wynik ($ELAPSED s)..."
done

echo "[7] Czyszczenie plików z klastra..."
ssh "$REMOTE_HOST" "rm -f $REMOTE_DIR/$REMOTE_INPUT $REMOTE_DIR/$REMOTE_OUTPUT $REMOTE_DIR/visualisations_script.py $REMOTE_DIR/slurm_job.sh $REMOTE_DIR/job.sh $REMOTE_DIR/*.out"

echo "Koniec"
