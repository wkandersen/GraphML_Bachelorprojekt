def compute_venue_homogeneity(paper_c_paper_edges, venue_value):
    """
    Computes venue homogeneity for each paper:
    fraction of citations where the cited paper has the same venue.

    Args:
        paper_c_paper_edges (DataFrame): Two-column DataFrame or Tensor containing paper-paper citations.
        venue_value (dict): Mapping from paper_id to venue_id.

    Returns:
        overall_avg_homogeneity: average across all papers
        per_paper_homogeneity: dict of paper_id → homogeneity score
    """
    from collections import defaultdict

    venue_value = {int(k): int(v) for k, v in venue_value.items()}

    # Build cited-by mapping
    citation_dict = defaultdict(list)
    for src, tgt in zip(paper_c_paper_edges[0], paper_c_paper_edges[1]):
        citation_dict[src.item()].append(tgt.item())

    per_paper_homogeneity = {}
    valid_paper_count = 0
    total_homogeneity = 0

    for paper, cited in citation_dict.items():
        paper_venue = venue_value.get(paper, None)
        if paper_venue is None or len(cited) == 0:
            continue

        match_count = 0
        total_count = 0
        for cited_paper in cited:
            cited_venue = venue_value.get(cited_paper, None)
            if cited_venue is not None:
                total_count += 1
                if cited_venue == paper_venue:
                    match_count += 1

        if total_count > 0:
            homogeneity = match_count / total_count
            per_paper_homogeneity[paper] = homogeneity
            total_homogeneity += homogeneity
            valid_paper_count += 1

    overall_avg = total_homogeneity / valid_paper_count if valid_paper_count > 0 else 0
    return overall_avg, per_paper_homogeneity
