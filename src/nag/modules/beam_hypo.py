import torch


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            # 可更新的情况：数量未饱和或超过最差得分
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                # 数量饱和需要删掉一个最差的
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        相关样本是否已经完成生成。
        best_sum_logprobs是新的候选序列中的最高得分。
        """
        if len(self) < self.num_beams:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # 是否最高分比当前保存的最低分还差
            ret = self.worst_score >= cur_score
            return ret


def beam_search(batch_size, num_beams, sos_token_id, ):
    beam_scores = torch.zeros((batch_size, num_beams))  # 定义scores向量，保存累加的log_probs
    beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
    beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
    done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty=0.7)
        for _ in range(batch_size)
    ]  # 为每个输入句子定义维护其beam search序列的类实例
    # 初始输入: （batch_size * num_beams, 1）个sos token
    input_ids = torch.full((batch_size * num_beams, 1), sos_token_id, dtype=torch.long)
    # h0: (1, batch_size * num_beams, hidden_size)
    hidden = torch.zeros((1, batch_size * num_beams, hidden_size))

    while cur_len < max_length:
        # outputs: (batch_size*num_beams, cur_len, vocab_size)
        outputs, hidden = decoder(input_ids, hidden)
        # 取最后一个timestep的输出 (batch_size*num_beams, vocab_size)
        next_token_logits = outputs[:, -1, :]
        scores = F.log_softmax(next_token_logits, dim=-1)  # log_softmax
        next_scores = scores + beam_scores[:, None].expand_as(scores)  # 累加上以前的scores
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
        # 取topk
        # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
        next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1, largest=True, sorted=True)

    next_batch_beam = []

    for batch_idx in range(batch_size):
        if done[batch_idx]:
            # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
            next_batch_beam.extend([(0, PAD_TOKEN, 0)] * num_beams)  # pad the batch
            continue
            next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
        for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
            zip(next_tokens[batch_idx], next_scores[batch_idx])
        ):
            beam_id = beam_token_id // vocab_size  # 1
            token_id = beam_token_id % vocab_size  # 1
            # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
            # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
            # batch_idx=1时，真实beam_id如下式计算为4或5
            effective_beam_id = batch_idx * num_beams + beam_id
            # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
            if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                if is_beam_token_worse_than_top_num_beams:
                    continue
                generated_hyps[batch_idx].add(
                    input_ids[effective_beam_id].clone(), beam_token_score.item(),
                )
            else:
                # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

            if len(next_sent_beam) == num_beams:
                break
            # 当前batch是否解码完所有句子
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )  # 注意这里取当前batch的所有log_prob的最大值
        # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
        # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
        next_batch_beam.extend(next_sent_beam)
        # 如果batch中每个句子的beam search都完成了，则停止
        if all(done):
            break
        # 准备下一次循环(下一层的解码)
        # beam_scores: (num_beams * batch_size)
        # beam_tokens: (num_beams * batch_size)
        # beam_idx: (num_beams * batch_size) 
        # 这里beam idx shape不一定为num_beams * batch_size，一般是小于等于
        # 因为有些beam id对应的句子已经解码完了 (下面假设都没解码完)
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])
        # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面, 
        # 因为有些beam id对应的句子已经解码完了
        input_ids = input_ids[beam_idx, :] # (num_beams * batch_size, seq_len)
        # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1
    # 注意有可能到达最大长度后，仍然有些句子没有遇到eos token，这时done[batch_idx]是false
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue
        for beam_id in range(num_beams):
            # 对于每个batch_idx的每句beam，都执行加入add
            # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)
    # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
    # 下面选择若干最好的序列输出
    # 每个样本返回几个句子
    output_num_return_sequences_per_batch = 1
    output_batch_size = output_num_return_sequences_per_batch * batch_size
    # 记录每个返回句子的长度，用于后面pad
    sent_lengths = input_ids.new(output_batch_size)
    best = []
    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        # x: (score, hyp), x[0]: score
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)
    if sent_lengths.min().item() != sent_lengths.max().item():
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        # fill pad
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)
        # 填充内容
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
            else:
                # 否则直接堆叠起来
                decoded = torch.stack(best).type(torch.long)
        # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
    return decoded
