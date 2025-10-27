import numpy as np
import evaluate
from collections import Counter, defaultdict
import math
from typing import List, Dict, Tuple
import torch
import transformers
import os

def calculate_ngram_coverage(
    generated_texts: List[str], train_texts: List[str], n: int = 3, word_based=True
) -> float:
    """
    Calculate n-gram coverage between generated and training texts.
    Higher coverage indicates better local fluency.

    Args:
        generated_texts: List of generated text samples
        train_texts: List of training text samples
        n: Size of n-grams to consider

    Returns:
        Coverage ratio (0-1)
    """
    if word_based:

        def get_ngrams(text: str, n: int) -> set:
            text = text.split()
            return {" ".join(text[i : i + n]) for i in range(len(text) - n + 1)}

    else:

        def get_ngrams(text: str, n: int) -> set:
            return {text[i : i + n] for i in range(len(text) - n + 1)}

    train_ngrams = set()
    for text in train_texts:
        train_ngrams.update(get_ngrams(text, n))

    ngram_coverage = []
    for text in generated_texts:
        gen_ngrams = set()
        gen_ngrams.update(get_ngrams(text, n))
        ngram_coverage.append(
            len(gen_ngrams.intersection(train_ngrams)) / len(gen_ngrams)
        )

    return np.mean(ngram_coverage)


def unique_ngrams_per_sample(
    generated_texts: List[str], n: int = 3, word_based=True
) -> float:
    if word_based:

        def get_ngrams(text: str, n: int) -> set:
            text = text.split()
            return {" ".join(text[i : i + n]) for i in range(len(text) - n + 1)}

    else:

        def get_ngrams(text: str, n: int) -> set:
            return {text[i : i + n] for i in range(len(text) - n + 1)}

    unique_ngrams = []
    for text in generated_texts:
        gen_ngrams = set()
        cur_ngrams = get_ngrams(text, n)
        gen_ngrams.update(get_ngrams(text, n))
        unique_ratio = len(gen_ngrams) / len(cur_ngrams)
        unique_ngrams.append(unique_ratio)
    return np.mean(unique_ngrams)


def total_unique_ngrams(
    generated_texts: List[str], n: int = 3, word_based=True
) -> float:
    if word_based:

        def get_ngrams(text: str, n: int) -> set:
            text = text.split()
            return {" ".join(text[i : i + n]) for i in range(len(text) - n + 1)}

    else:

        def get_ngrams(text: str, n: int) -> set:
            return {text[i : i + n] for i in range(len(text) - n + 1)}

    gen_ngrams = set()
    total_ngrams = 0
    for text in generated_texts:
        cur_ngrams = get_ngrams(text, n)
        total_ngrams += len(cur_ngrams)
        gen_ngrams.update(get_ngrams(text, n))

    unique_ratio = len(gen_ngrams) / total_ngrams
    return unique_ratio


def calculate_char_distribution_metrics(
    generated_texts: List[str], train_texts: List[str]
) -> Tuple[Dict[str, float], float]:
    """
    Calculate character frequency distributions and KL divergence.

    Args:
        generated_texts: List of generated text samples
        train_texts: List of training text samples

    Returns:
        Tuple of (char_freqs, kl_divergence)
    """

    def get_char_freqs(texts: List[str]) -> Dict[str, float]:
        all_chars = "".join(texts)
        total_chars = len(all_chars)
        freqs = Counter(all_chars)
        return {char: count / total_chars for char, count in freqs.items()}

    train_freqs = get_char_freqs(train_texts)
    gen_freqs = get_char_freqs(generated_texts)

    # Calculate KL divergence
    kl_div = 0.0
    for char in train_freqs:
        if char in gen_freqs:
            kl_div += train_freqs[char] * math.log(train_freqs[char] / gen_freqs[char])

    return gen_freqs, kl_div


def calculate_word_dictionary_match(
    generated_texts: List[str], train_texts: List[str], n: int = 4
) -> float:
    """
    Calculate the ratio of words in generated text that appear in training vocabulary.

    Args:
        generated_texts: List of generated text samples
        train_texts: List of training text samples

    Returns:
        Ratio of matching words (0-1)
    """
    train_words = set()
    train_file = (
        "/home/patrick/.cache/discrete_diffusion/text8/text8/raw_data/text8.test.txt"
    )
    # Split by whitespace into words
    with open(train_file, "r") as f:
        text = f.read().lower()  # Ensure it's lowercase like the dataset expects

    tokens = text.split()

    # Count word frequencies
    word_counts = Counter(tokens)

    # Filter out 1-letter words
    filtered_vocab = {word for word in word_counts if len(word) > n}

    total = 0
    word_counts_total = []
    for l in generated_texts:
        seen = []
        counter = 0
        for w in l.split():
            counter += 1
            seen.append(w)

        gen_words = set(seen)
        matching_words = gen_words.intersection(filtered_vocab)
        wc = len(matching_words) / counter
        word_counts_total.append(wc)

    with open("/home/patrick/bd3lms/word_counts_total.txt", "w") as f:
        for wc in word_counts_total:
            f.write(f"{wc}\n")

    return np.mean(word_counts_total)


def total_valid_words(
    generated_texts: List[str], train_texts: List[str], n: int = 4
) -> float:
    """
    Calculate the ratio of words in generated text that appear in training vocabulary.

    Args:
        generated_texts: List of generated text samples
        train_texts: List of training text samples

    Returns:
        Ratio of matching words (0-1)
    """
    train_words = set()
    train_file = (
        "/home/patrick/.cache/discrete_diffusion/text8/text8/raw_data/text8.test.txt"
    )
    # Split by whitespace into words
    with open(train_file, "r") as f:
        text = f.read().lower()  # Ensure it's lowercase like the dataset expects

    tokens = text.split()

    # Count word frequencies
    word_counts = Counter(tokens)

    # Filter out 1-letter words
    filtered_vocab = {word for word in word_counts if len(word) > 1}

    total = 0
    word_counts_total = []
    for l in generated_texts:
        seen = []
        counter = 0
        for w in l.split():
            seen.append(w)
            if w in filtered_vocab:
                counter += 1
        valid_pct = counter / len(seen)
        word_counts_total.append(valid_pct)
    return np.mean(word_counts_total)


def total_unique_words(
    generated_texts: List[str], train_texts: List[str], n: int = 4
) -> float:
    """
    Calculate the ratio of words in generated text that appear in training vocabulary.

    Args:
        generated_texts: List of generated text samples
        train_texts: List of training text samples

    Returns:
        Ratio of matching words (0-1)
    """

    total = 0
    word_counts_total = []
    for l in generated_texts:
        seen = []
        counter = 0
        for w in l.split():
            seen.append(w)
        unique_pct = len(set(seen)) / len(seen)
        word_counts_total.append(unique_pct)
    return np.mean(word_counts_total)


def calculate_repetition_rate(texts: List[str], n: int = 4) -> float:
    """
    Calculate the rate of repeated n-grams in the generated text.
    Higher values indicate potential overfitting or mode collapse.

    Args:
        texts: List of text samples
        n: Size of n-grams to consider

    Returns:
        Repetition rate (0-1)
    """

    def get_ngram_counts(text: str, n: int, word_based=True) -> Dict[str, int]:
        counts = defaultdict(int)
        if word_based:
            text = text.split()
            for i in range(len(text) - n + 1):
                counts[" ".join(text[i : i + n])] += 1
        else:
            for i in range(len(text) - n + 1):
                counts[text[i : i + n]] += 1
        return counts

    total_ngrams = 0
    repeated_ngrams = 0

    for text in texts:
        counts = get_ngram_counts(text, n)
        total_ngrams += len(counts)
        repeated_ngrams += sum(1 for count in counts.values() if count > 1)

    return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0


def record_generative_perplexity(
    text_samples: List[str],
    model_name_or_path: str = "gpt2-large",
    batch_size: int = 4,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Calculate the generative perplexity of text samples using a pretrained model.
    Returns per-sample perplexity statistics.

    Args:
        text_samples: List of generated text samples
        model_name_or_path: Name of pretrained model to use for evaluation (default: "gpt2-large")
        batch_size: Batch size for processing (default: 4)
        device: Device to run the model on (default: "cuda")

    Returns:
        Dictionary containing perplexity statistics (mean, std, min, max)
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path).eval()

    # Set padding token if none
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Move model to device
    model = model.to(device)

    # Tokenize text samples
    encodings = tokenizer(
        text_samples, return_tensors="pt", padding=True, truncation=True
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Calculate perplexity in batches
    sample_nlls = []  # Store per-sample NLLs

    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_attention_mask = attention_mask[i : i + batch_size]

        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits

            # Calculate loss for each token (shifted)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_input_ids[..., 1:].contiguous()
            shift_attention_mask = batch_attention_mask[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # Apply attention mask to loss
            loss = loss.view(shift_labels.size())
            loss = loss * shift_attention_mask

            # Calculate NLL per sample (average over tokens)
            nll_per_sample = loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
            sample_nlls.extend(nll_per_sample.cpu().tolist())

    # Convert NLLs to perplexities
    sample_perplexities = [torch.exp(torch.tensor(nll)).item() for nll in sample_nlls]

    # Calculate statistics
    perplexity_stats = {
        "perplexity_mean": float(np.mean(sample_perplexities)),
        "perplexity_std": float(np.std(sample_perplexities)),
        "perplexity_min": float(np.min(sample_perplexities)),
        "perplexity_max": float(np.max(sample_perplexities)),
        # 'perplexity_per_sample': sample_perplexities
    }

    return perplexity_stats


def compute_self_bleu_hf(sentences: List[str], n_gram: int = 4) -> float:
    """
    Compute Self-BLEU using Hugging Face's evaluate library.
    Args:
        sentences: list of text strings
        n_gram: BLEU n-gram order
    Returns:
        Mean Self-BLEU score
    """
    bleu_metric = evaluate.load("bleu")
    scores = []

    for i in range(len(sentences)):
        hypothesis = sentences[i]
        references = sentences[:i] + sentences[i + 1 :]
        result = bleu_metric.compute(
            predictions=[hypothesis], references=[references], max_order=n_gram
        )
        scores.append(result["bleu"])

    return float(np.mean(scores))


def evaluate_text_metrics(
    generated_texts: List[str],
    train_texts: List[str],
    *args, 
    **kwargs,
) -> Dict[str, float]:
    """
    Calculate all text metrics for generated samples.

    Args:
        generated_texts: List of generated text samples
        train_texts: List of training text samples
        include_perplexity: Whether to include perplexity metrics (computationally expensive)
        perplexity_model: Model to use for perplexity calculation if included

    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        "ngram_coverage_3": calculate_ngram_coverage(generated_texts, train_texts, n=3),
        "ngram_coverage_4": calculate_ngram_coverage(generated_texts, train_texts, n=4),
        "unique_ngrams_per_sample_3": unique_ngrams_per_sample(generated_texts, n=3),
        "unique_ngrams_per_sample_4": unique_ngrams_per_sample(generated_texts, n=4),
        "total_unique_ngrams_3": total_unique_ngrams(generated_texts, n=3),
        "total_unique_ngrams_4": total_unique_ngrams(generated_texts, n=4),
        "repetition_rate_3": calculate_repetition_rate(generated_texts, n=3),
        "repetition_rate_4": calculate_repetition_rate(generated_texts, n=4),
        "total_valid_words": total_valid_words(generated_texts, train_texts),
        "total_unique_words": total_unique_words(generated_texts, train_texts),
    }
    return metrics
