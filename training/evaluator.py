# Adapted from https://github.com/facebookresearch/SymbolicMathematics under CC BY-NC 4.0

"""
Evaluator class used the check the validity of our solution
"""

from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import sympy as sp

from add_ons.mathematica_utils import sp_to_mma, check_numerical_equiv
from environment.utils import to_cuda, timeout, TimeoutError
from environment.char_env import InvalidPrefixExpression

logger = getLogger()

BUCKET_LENGTH_SIZE = 5


def idx_to_sp(env, idx, return_infix=False, return_info=False):
    """
    Convert an indexed prefix expression to SymPy.
    """
    prefix = [env.id2word[wid] for wid in idx]
    try:
        infix = env.prefix_to_infix(prefix)
        if return_info:
            prefix_info = prefix[prefix.index('&'):]
            info_infix = env.scr_prefix_to_infix(prefix_info)

    except InvalidPrefixExpression:
        return None

    # In case the parser fails (e.g for expressions that are too long)
    try:
        eq = sp.S(infix, locals=env.local_dict)
    except :
        return None
    if return_info:
        eq = [eq, info_infix]
    return (eq, infix) if return_infix else eq


@timeout(5000)
def check_valid_solution(env, src, tgt, hyp, session):
    """
    Check that a solution is valid.
    """

    if env.save_info_scr:
        tgt = tgt[0]

    if env.numerical_check:
        # Pre check symbolically
        valid = sp.simplify(hyp - tgt, seconds=0.5) == 0

        if session is None:
            raise ValueError('Session should not be None to numerically evaluate')

        # Do the numerical check
        if not valid:
            hyp_mma = sp_to_mma(hyp, env.bracket_tokens, env.func_dict)
            tgt_mma = sp_to_mma(tgt, env.bracket_tokens, env.func_dict)
            valid, _ = check_numerical_equiv(session, hyp_mma, tgt_mma)
            if valid:
                logger.info("Hypothesis is numerically valid")
    else:
        valid = sp.simplify(hyp - tgt, seconds=5) == 0
    return valid


@timeout(5000)
def check_hypothesis(eq, session):
    """
    Check a hypothesis for a given equation and its solution.
    """
    env = Evaluator.ENV
    src = idx_to_sp(env, eq['src'])
    tgt = idx_to_sp(env, eq['tgt'], return_info=env.save_info_scr)
    hyp = eq['hyp']

    if env.save_info_scr:
        if env.word2id['&'] not in eq['hyp']:
            eq['src'] = str(src)
            eq['tgt'] = tgt
            eq['hyp'] = 'No & delimiter'
            eq['is_valid'] = False
            return eq
        else:
            hyp_infos = hyp[hyp.index(env.word2id['&']):]
            hyp_infos_pre = [env.id2word[wid] for wid in hyp_infos]
            hyp = hyp[:hyp.index(env.word2id['&'])]

    hyp_infix = [env.id2word[wid] for wid in hyp]

    try:
        hyp, hyp_infix = idx_to_sp(env, hyp, return_infix=True)
        is_valid = check_valid_solution(env, src, tgt, hyp, session)
        hyp_infix = str(hyp)

        if env.save_info_scr:
            hyp_infix += env.scr_prefix_to_infix(hyp_infos_pre)

    except (TimeoutError, Exception) as e:
        e_name = type(e).__name__
        if not isinstance(e, InvalidPrefixExpression):
            logger.error(f"Exception {e_name} when checking hypothesis: {hyp_infix}")
        hyp = f"ERROR {e_name}"
        is_valid = False

    # update hypothesis
    eq['src'] = str(src)
    eq['tgt'] = tgt
    eq['hyp'] = hyp_infix
    eq['is_valid'] = is_valid

    return eq


class Evaluator(object):
    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        self.session = None
        Evaluator.ENV = trainer.env

    def add_mathematica_session(self, session):
        self.session = session

    def run_all_evals(self):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': self.trainer.epoch})

        # save statistics about generated data
        if self.params.export_data:
            scores['total'] = sum(self.trainer.EQUATIONS.values())
            scores['unique'] = len(self.trainer.EQUATIONS)
            scores['unique_prop'] = 100. * scores['unique'] / scores['total']
            return scores

        with torch.no_grad():
            for data_type in ['valid', 'test']:
                for task in self.params.tasks:
                    if self.params.beam_eval:
                        self.enc_dec_step_beam(data_type, task, scores)
                    else:
                        self.enc_dec_step(data_type, task, scores)

        return scores

    def enc_dec_step(self, data_type, task, scores):
        """
        Encoding / decoding step.
        """
        params = self.params
        env = self.env
        encoder, decoder = self.modules['encoder'], self.modules['decoder']
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ['spin_hel']

        # stats
        xe_loss = 0
        n_valid = torch.zeros(1000, dtype=torch.long)
        n_total = torch.zeros(1000, dtype=torch.long)

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(params.dump_path, f"eval.{task}.{scores['epoch']}")
            f_export = open(eval_path, 'w')
            logger.info(f"Writing evaluation results in {eval_path} ...")

        # iterator
        iterator = self.env.create_test_iterator(data_type, task, params=params, data_path=self.trainer.data_path)
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # print status
            if n_total.sum().item() % 100 < params.batch_size:
                logger.info(f"{n_total.sum().item()}/{eval_size}")

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

            # forward / loss
            encoded = encoder('fwd', x=x1, lengths=len1, causal=False)
            decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1)
            word_scores, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True)

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()

            # export evaluation details
            if params.eval_verbose:
                for i in range(len(len1)):
                    src = idx_to_sp(env, x1[1:len1[i] - 1, i].tolist())
                    tgt = idx_to_sp(env, x2[1:len2[i] - 1, i].tolist())
                    s = f"Equation {n_total.sum().item() + i} ({'Valid' if valid[i] else 'Invalid'})\nsrc={src}\ntgt={tgt}\n"
                    if params.eval_verbose_print:
                        logger.info(s)
                    f_export.write(s + "\n")
                    f_export.flush()

            # stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

        # evaluation details
        if params.eval_verbose:
            f_export.close()

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations were evaluated correctly.")

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size or self.trainer.data_path
        scores[f'{data_type}_{task}_xe_loss'] = xe_loss / _n_total
        scores[f'{data_type}_{task}_acc'] = 100. * _n_valid / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            # logger.info(f"{i}: {n_valid[i].item()} / {n_total[i].item()} ({100. * n_valid[i].item() / max(n_total[i].item(), 1)}%)")
            scores[f'{data_type}_{task}_acc_{i}'] = 100. * n_valid[i].item() / max(n_total[i].item(), 1)

    def enc_dec_step_beam(self, data_type, task, scores):
        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        params = self.params
        env = self.env
        encoder, decoder = self.modules['encoder'], self.modules['decoder']
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1, 2]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ['spin_hel']

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(params.dump_path, f"eval.{task}.{data_type}.{scores['epoch']}")
            f_export = open(eval_path, 'w')
            logger.info(f"Writing evaluation results in {eval_path} ...")

        def display_logs(logs, offset):
            """
            Display detailed results about success / fails.
            """
            if params.eval_verbose == 0:
                return
            for i, res in sorted(logs.items()):
                if env.save_info_scr:
                    tgt_str = str(res['tgt'][0]) + ' IDS : ' + res['tgt'][1]
                else:
                    tgt_str = str(res['tgt'])
                n_valid = sum([int(v) for _, _, v in res['hyps']])
                s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\nsrc={res['src']}\ntgt={tgt_str}\n"
                for hyp, score, valid in res['hyps']:
                    if score is None:
                        s += f"{int(valid)} {hyp}\n"
                    else:
                        s += f"{int(valid)} {score :.3e} {hyp}\n"
                if params.eval_verbose_print:
                    logger.info(s)
                f_export.write(s + "\n")
                f_export.flush()

        # stats
        xe_loss = 0
        n_valid = torch.zeros(1000, params.beam_size, dtype=torch.long)
        n_total = torch.zeros(1000, dtype=torch.long)

        # iterator
        iterator = env.create_test_iterator(data_type, task, params=params, data_path=self.trainer.data_path)
        eval_size = len(iterator.dataset)

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
            bs = len(len1)

            # forward
            encoded = encoder('fwd', x=x1, lengths=len1, causal=False)
            decoded = decoder('fwd', x=x2, lengths=len2, causal=True, src_enc=encoded.transpose(0, 1), src_len=len1)
            word_scores, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True)

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()

            # save evaluation details
            beam_log = {}
            for i in range(len(len1)):
                src = idx_to_sp(env, x1[1:len1[i] - 1, i].tolist())
                if src is None:
                    src = 'Invalid prefix expression'
                tgt = idx_to_sp(env, x2[1:len2[i] - 1, i].tolist(), return_info=env.save_info_scr)

                if valid[i]:
                    beam_log[i] = {'src': src, 'tgt': tgt, 'hyps': [(tgt, None, True)]}

            # stats
            xe_loss += loss.item() * len(y)
            n_valid[:, 0].index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

            # continue if everything is correct. if eval_verbose, perform
            # a full beam search, even on correct greedy generations
            if valid.sum() == len(valid) and params.eval_verbose < 2:
                display_logs(beam_log, offset=n_total.sum().item() - bs)
                continue

            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(
                f"({n_total.sum().item()}/{eval_size}) Found {bs - len(invalid_idx)}/{bs} valid top-1 predictions. Generating solutions ...")

            try:
                # generate
                _, _, generations = decoder.generate_beam(
                    encoded.transpose(0, 1),
                    len1,
                    beam_size=params.beam_size,
                    length_penalty=params.beam_length_penalty,
                    early_stopping=params.beam_early_stopping,
                    max_len=params.max_len
                )
            except (TimeoutError, Exception) as e:
                _, _, generations = decoder.generate_beam(
                    encoded.transpose(0, 1),
                    len1,
                    beam_size=1,
                    length_penalty=params.beam_length_penalty,
                    early_stopping=params.beam_early_stopping,
                    max_len=params.max_len
                )
                generations[0].n_hyp = params.beam_size
                generations[0].hyp.extend([generations[0].hyp for _ in range(params.beam_size - 1)])

            # prepare inputs / hypotheses to check
            # if eval_verbose < 2, no beam search on equations solved greedily

            inputs = []
            for i in range(len(generations)):
                if valid[i] and params.eval_verbose < 2:
                    continue
                for j, (score, hyp) in enumerate(sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)):
                    inputs.append({
                        'i': i,
                        'j': j,
                        'score': score,
                        'src': x1[1:len1[i] - 1, i].tolist(),
                        'tgt': x2[1:len2[i] - 1, i].tolist(),
                        'hyp': hyp[1:].tolist(),
                    })

            # check hypotheses
            outputs = []
            for input_eq in inputs:
                try:
                    outputs.append(check_hypothesis(input_eq, self.session))
                except (TimeoutError, Exception) as e:
                    outputs.append(outputs[-1])

            # read results
            for i in range(bs):

                # select hypotheses associated to current equation
                gens = sorted([o for o in outputs if o['i'] == i], key=lambda x: x['j'])
                assert (len(gens) == 0) == (valid[i] and params.eval_verbose < 2) and (i in beam_log) == valid[i]
                if len(gens) == 0:
                    continue

                # source / target
                src = gens[0]['src']
                tgt = gens[0]['tgt']
                beam_log[i] = {'src': src, 'tgt': tgt, 'hyps': []}

                # for each hypothesis
                for j, gen in enumerate(gens):

                    # sanity check
                    assert gen['src'] == src and gen['tgt'] == tgt and gen['i'] == i and gen['j'] == j

                    # if the hypothesis is correct, and we did not find a correct one before
                    is_valid = gen['is_valid']
                    if is_valid and not valid[i]:
                        n_valid[nb_ops[i], j] += 1
                        valid[i] = 1

                    # update beam log
                    beam_log[i]['hyps'].append((gen['hyp'], gen['score'], is_valid))

            # valid solutions found with beam search
            logger.info(f"    Found {valid.sum().item()}/{bs} solutions in beam hypotheses.")

            # export evaluation details
            if params.eval_verbose:
                assert len(beam_log) == bs
                display_logs(beam_log, offset=n_total.sum().item() - bs)

        # evaluation details
        if params.eval_verbose:
            f_export.close()
            logger.info(f"Evaluation results written in {eval_path}")

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        logger.info(f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) equations were evaluated correctly.")
        logger.info(n_valid[
                    :(n_valid.sum(1) > 0).nonzero().view(-1)[-1] + 1,
                    :(n_valid.sum(0) > 0).nonzero().view(-1)[-1] + 1
                    ])

        # compute perplexity and prediction accuracy
        assert _n_total == eval_size or self.trainer.data_path
        scores[f'{data_type}_{task}_beam_acc'] = 100. * _n_valid / _n_total

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            logger.info(
                f"{i}: {n_valid[i].sum().item()} / {n_total[i].item()} ({100. * n_valid[i].sum().item() / max(n_total[i].item(), 1)}%)")
            scores[f'{data_type}_{task}_beam_acc_{i}'] = 100. * n_valid[i].sum().item() / max(n_total[i].item(), 1)


def convert_to_text(batch, lengths, id2word, params):
    """
    Convert a batch of sequences to a list of text sequences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sequences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(id2word[batch[k, j]])
        sequences.append(" ".join(words))
    return sequences
