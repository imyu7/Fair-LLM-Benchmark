# conda activate bias
export PYTHONPATH=.

# Evaluate all datasets
for dataset in "BBQ" "Equity-Evaluation-Corpus" "WinoBias"
do
    echo "----Evaluating $dataset----------------"
    python $dataset/src/evaluation.py
done
