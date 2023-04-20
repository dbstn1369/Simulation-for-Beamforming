def calculate_state_variables(STS, SSW, BI, prev_BI=None):
    # 현재 STS 수 계산
    s_sts = len(STS)

    # 전파 지연 계산
    propagation_delay = []
    for sts in STS:
        delay = np.linalg.norm(sts.position - SSW.position) / 3e8  # 전파속도
        propagation_delay.append(delay)
    s_pd = np.mean(propagation_delay)

    # 혼잡도 계산
    s_c = np.mean([ssw.snr for ssw in SSW.ssw_list])

    # STS 사용 모델 계산
    num_ssw = sum([len(sts.ssw_list) for sts in STS])
    if prev_BI is not None:
        prev_num_ssw = sum([len(sts.ssw_list) for sts in prev_BI.STS])
        delta_u_norm = (num_ssw - prev_num_ssw) / prev_num_ssw if prev_num_ssw != 0 else 0
    else:
        delta_u_norm = 0
    s_u = num_ssw / len(STS)

    return s_sts, s_pd, s_c, s_u, delta_u_norm