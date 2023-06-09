# TRL库实现代码的理解

"""
# PPO计算流程
- 通过actor_mode计算每个位置的 logprobs 和 values
- 通过ref_model计算每个位置的 ref_logprobs
- 通过reward_model计算每句话的总分 scores

- 计算每个位置的即时rewards:r
  - 用到 logprobs和ref_logprobs计算KL
  - 将scores加到最后一个位置
  
- train-minibatch-ppo
  - 计算pg_loss
    - 利用rewards和values计算advantage
    - advantage.detach
    - 重新计算当前正在训练的actor得到当前步的logprobs和values
    - importance-sampling得到ratio，用到的old_logprobs是流程一开始得到的logprobs
    - 得到pg_loss

  - 计算value_loss
    - 当前的 (values - (advantage + old_values))**2
      - MSE loss
      - 这里用到的old_values也是流程一开始得到的values


# 整体流程
- 每个epoch：
  - 将当前的actor视为未来一段实际内ppo-minibatch训练期间实际与环境交互的模型，
  - 采样128个prompt作为query，用actor生成128个response，计算log_prob等信息
    - 每个ppo-epoch：
      - 每次使用一个完整句子的相关信息，做minibatch-ppo训练，更新actor参数
      - 在每次迭代计算的时候，importance-sampling用到的old_logprob，就是上面的log_prob，是不变的
    - 结束本轮ppo迭代后，当前actor的参数被更新
  - 当前actor已经被更新，继续做epoch时，相当于真正的跟环境交互的模型也更新了。



# 疑问
- value_loss的计算公式不太明白
  - value_loss 就应该等于critic产生的预测值 v_pred 和真实值 r + v_next 之间的差值
  - 这个真实值怎么理解，它不也就是上一轮中的critic预测的值吗这里的v_next也是一个变量呀，难度是因为用r做了修正，我们希望更新critic，使得他的输出更符合v = r + v_next

- 计算每个位置的reward，为什么最后一个位置是句子的score，其他位置是KL-penalty


"""




import torch



class PPOTrainer:
    

    def step(self, queries, responses, scores):
        """
        Run a PPO optimisation step.

        args:
            queries (List): List of tensors containing the encoded queries, shape [query_length]
            responses (List): List of tensors containing the encoded responses, shape [response_length]
            scores (List): tensor containing the scores, shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """
        #[x]拿到actor_model和ref_model输出的目标token的log_prob，以及critic输出的每个位置的value
        logprobs, ref_logprobs, values = self.batched_forward_pass(queries, responses) #每项都是128个一维张量组成的列表：shape: (gen_len,)

        #[x]计算每个token的即时reward:r  
        rewards = self.compute_rewards(scores, logprobs, ref_logprobs) #128个一维张量组成的列表: shape: (gen_len,)  

        #一批采样数据会进行多轮的ppo训练
        for _ in range(self.ppo_params['ppo_epochs']):
            #[x]每次随机采样一个句子相关的信息做ppo训练
            for i in range(bs):
                train_stats = self.train_minibatch(logprobs[idx].unsqueeze(0), 
                                                   values[idx].unsqueeze(0),
                                                   rewards[idx].unsqueeze(0), 
                                                   responses[idx].unsqueeze(0),
                                                   torch.cat([queries[idx],responses[idx]]).unsqueeze(0))



    def batched_forward_pass(self, queries, responses): #response是actor输出的答案
        """
        使用query和标准response，分别送到actor_model和ref_model，得到：
            - 两个模型在每个位置对正确token的预测log_prob
            - critic_model预测的每个位置的value（注意这里critic_model用的是actor_model的value_head预测结果，原因在下面importance-sampling部分）
        returns:
            all_logprobs, all_ref_logprobs, all_values
        """
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        input_ids # 拼接queries和reponses每项，注意这里的response部分是actor_model输出的，看作是标准答案。#[x]这样看的话，按理说也可以是人准备的标准答案呀，但是和原始模型差别太大不知道行不行
        with torch.no_grad():
            #[ ]应用模型forward，计算每个位置的预测token，注意这里不是generate往后继续预测
            logits, _, v = self.model(input_ids)      # logits -> (batch, seq_len, vocab_size); v -> (batch, seq_len) 
            ref_logits, _, _ = self.ref_model(input_ids)  #注意这里的model和ref_model都是添加了Value_head的，输出的v：returns a scalar for each output token

        logprobs = logprobs_from_logits(logits[:,:-1,:], input_ids[:,1:])   # (batch, seq_len-1) 每一个元素是句子当前位置应该预测的正确token的 log_probablity
        ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], input_ids[:,1:])   # (batch, seq_len - 1)
        
        #[ ]拿到模型生成部分的结果（去掉prompt部分）
        for j in range(fbs):
            start = len(query_batch[j]) - 1
            end = start + len(response_batch[j])
            all_values.append(v[j, start:end])    #[ ]                            # 生成的每个token的value,  shape:[16]
            all_logprobs.append(logprobs[j, start:end])                           # 生成的每个token的log_prob, shape:[16]
            all_ref_logprobs.append(ref_logprobs[j, start:end])                   # ref_model生成的每个token的log_prob, shape: [16]
        return all_logprobs, all_ref_logprobs, all_values



    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token reward from scores and KL-penalty."""
        rewards = []
        #[x]根据每个位置的logprob和ref_logprob得到KL-pelnaty，再将整个句子的Reward加到最后一个位置，得到每个位置的rewards
        for score, logprob, ref_logprob in zip(scores, logprobs, ref_logprobs):  # score: [1]; logprob:[16]; ref_logprobs:[16]
            kl = logprob - ref_logprob    #kl = log(p/q) = log_p - log_q     # (gen_len, )  #因为我们计算的是采取action的KL，所以在每个位置计算的不是两个模型完整预测词表的概率分布的KL，而是将ref_model视为一个标准，让actor预测的token概率尽量和ref_model靠近，所以只看该token的预测概率
            non_score_reward = -self.kl_ctl.value * kl    #对每个位置：actor预测的token和ref_model越靠近，是应该被鼓励的，reward相对应该越大。    # (gen_len, )
            reward = non_score_reward.clone()   # 前面每一个token的reward都来自KL惩罚
            reward[-1] += score               #只在最后一位加上整个句子的得分total Reward
            rewards.append(reward)
        return rewards                        # 128个 (gen_len, )


    def train_minibatch(self, logprobs, values, rewards, response, model_input):  #相当于一次完整的路径采样的结果
        """Train one PPO minibatch"""
        loss_p, loss_v, train_stats  = self.loss(logprobs, values, rewards, response, model_input) #loss_policy, loss_value
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return train_stats


    def loss(self, old_logprobs, values, rewards, response, model_input):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response.shape[1]

        #[x]利用rewards和values，从后往前计算每个位置的Advantage = r + V_t+1 - V_t，Advantage实际表示的依然是每个位置一直往后的累积total Reward的相对值，为什么是相对，因为减掉了期望
        for t in reversed(range(gen_len)):    
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0   #这里的values是critic算出来的，在该项目中，用的是actor model的value_head输出的值，也可以是一个独立的model
            delta = rewards[:, t] + self.ppo_params['gamma'] * nextvalues - values[:, t]   # TD error，得到的是advantage，这里的reward对应公式中的小r，是每个token的即时reward，它只在最后一个位置是total Reward，其他前面的位置都是-KL，我的理解是在这里从后往前计算的过程中，以及GAE的作用，total Reward会往前传递的？？
            lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam  # GAE 用于平衡 bias 和 variance， 可以理解为一种衰减移动平均
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1) #得到每个位置的advantage

        returns = advantages + values          #用于下方计算critic的value_loss      # (batch, generated_seq_len)
        advantages = whiten(advantages)
        
        #[x]丢弃Advantage的梯度关系
        advantages = advantages.detach()       #公式中advantage是作为一个常数乘在梯度里的，但是得到它的过程是建立了梯度关系的，所以要detach掉，这也意味着有些量需要重新进行前向计算，建立梯度图，然后才能根据Loss计算进行BP

        #[x]重新前向计算，获得参数的梯度关系 
        logits, _, vpred = self.model(model_input)          # logits -> (batch, all_seq_len, vocab_size); vpred -> (batch, all_seq_len)
        logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])

        #only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]  # logprob -> (batch, gen_len); vpred -> (batch, gen_len)

        #[x]pg_loss部分计算
        ratio = torch.exp(logprob - old_logprobs)   #[ ]从这里可以看出来，ppo训练过程中是将前一个阶段的actor视为与环境交互的q分布，难怪最开始采样response的时候用的是actor，包括critic的输出用的也是actor的value_head，所谓的reference_model仅用于计算KL，自始至终无更新。
        pg_losses = -advantages * ratio    # importance sampling
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.ppo_params['cliprange'],
                                               1.0 + self.ppo_params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2)) #[ ]得到pg_loss,  这里为什么是max，按李宏毅图看应该是min呀，可能是因为加了负号
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())


        #[x]value_loss计算，等于 Value Head 产生的预测值 v_pred 和真实值 r + v_next 之间的差值
        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])

        vf_losses1 = (vpred - returns)**2                # v - (r + v_next - v + v)
        vf_losses2 = (vpredclipped - returns)**2         # value loss clipped
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())


    
        loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

        return pg_loss, self.ppo_params['vf_coef'] * vf_loss, train_stats



