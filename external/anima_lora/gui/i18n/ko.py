"""Korean strings for the Anima LoRA GUI."""

from __future__ import annotations

STRINGS: dict[str, str] = {
    # Window / tabs
    "window_title": "Anima LoRA",
    "tab_config": "학습 설정",
    "tab_ip_adapter": "IP-Adapter",
    "tab_easycontrol": "EasyControl",
    "tab_postfix": "Postfix",
    "tab_methods": "메소드",
    "tab_images": "데이터셋",
    "tab_merge": "병합",
    "tab_preprocess": "전처리",
    # PreprocessingTab
    "preprocess_intro": (
        "캡션 셔플과 말풍선 마스킹을 설정하고, 각 단계를 원할 때 실행합니다. "
        "학습 설정 탭의 학습 버튼은 캐시가 없을 때 기본값으로 전처리를 "
        "자동 실행합니다 — 이 탭은 세부 조정 및 단계별 재실행용입니다."
    ),
    "preprocess_text_caching": "캐싱 (VAE + 텍스트)",
    "preprocess_caption_shuffle_variants": "캡션당 셔플 변형 수 (N):",
    "preprocess_caption_shuffle_variants_tip": (
        "이미지당 N개의 캡션 변형을 생성합니다. v0은 원본 그대로이고, "
        "v1..v(N-1)은 스마트 셔플되며 (태그 드롭아웃 > 0이면) @artist 이후 "
        "태그가 독립적으로 드롭됩니다. use_shuffled_caption_variants=true일 때 "
        "데이터로더는 v0을 20% 확률로, 나머지는 v1..v(N-1) 균등 분포로 선택합니다. "
        "0으로 설정하면 원본 캡션 하나만 캐싱합니다."
    ),
    "preprocess_caption_tag_dropout_rate": "태그 드롭아웃 비율 (0.0–1.0):",
    "preprocess_caption_tag_dropout_rate_tip": (
        "v1..v(N-1)에만 적용되는 태그별 드롭아웃 확률입니다. "
        "첫 번째 @artist 마커까지의 태그는 절대 드롭되지 않습니다. "
        "셔플 변형 수가 0 이하이면 무시됩니다."
    ),
    "preprocess_run_te": "캐싱 실행 (VAE + 텍스트)",
    "preprocess_masking_sam": "SAM3 마스킹 (말풍선)",
    "preprocess_masking_mit": "MIT 마스킹 (만화 텍스트)",
    "preprocess_sam_prompts": "SAM 프롬프트 (한 줄에 하나):",
    "preprocess_sam_prompts_tip": (
        "SAM3이 찾을 텍스트 프롬프트. 한 줄에 하나씩. "
        "기본값: 'speech bubble', 'text bubble'."
    ),
    "preprocess_sam_threshold": "SAM 임계값 (0.0–1.0):",
    "preprocess_sam_threshold_tip": (
        "SAM3 탐지를 유지할 최소 신뢰도. 낮을수록 더 많은 마스크 "
        "(오탐 포함 가능), 높을수록 엄격. 기본값 0.5."
    ),
    "preprocess_dilate": "팽창 (px):",
    "preprocess_dilate_tip": (
        "이진 마스크에 적용할 팽창 픽셀 수. 값이 클수록 마스크 가장자리가 "
        "바깥으로 번집니다. 기본값 5. 0으로 비활성화."
    ),
    "preprocess_mit_threshold": "MIT 텍스트 임계값 (0.0–1.0):",
    "preprocess_mit_threshold_tip": (
        "MIT/ComicTextDetector 텍스트 세그멘터의 신뢰도 임계값. 기본값 0.8."
    ),
    "preprocess_run_mask": "마스킹 실행",
    "preprocess_run_sam_mask": "SAM 마스킹 실행",
    "preprocess_run_sam_mask_tip": (
        "마스크 생성 단계에서 SAM3 말풍선 분할을 실행합니다. "
        "체크 해제하면 SAM을 건너뛰고 MIT(또는 활성화된 다른 백엔드)만 사용합니다."
    ),
    "preprocess_run_mit_mask": "MIT 마스킹 실행",
    "preprocess_run_mit_mask_tip": (
        "마스크 생성 단계에서 MIT/ComicTextDetector 텍스트 분할을 "
        "실행합니다. 체크 해제하면 MIT를 건너뛰고 SAM만 사용합니다."
    ),
    "preprocess_mask_nothing_enabled": (
        "SAM 또는 MIT 마스킹 중 최소 하나는 활성화되어야 합니다."
    ),
    "preprocess_status_resized": "리사이즈된 이미지: {n}장",
    "preprocess_status_caches": "캐시 — latents: {lat}, text: {te}, PE: {pe}",
    "preprocess_status_masks": "마스크: {masks}장",
    "preprocess_status_no_resized": "리사이즈된 이미지가 없습니다 — 학습 설정 탭에서 Preprocess를 먼저 실행하세요.",
    "preprocess_log_placeholder": "전처리 출력이 여기에 표시됩니다...",
    "preprocess_save_settings": "저장",
    "preprocess_save_settings_tip": "이 설정들을 디스크에 저장합니다 (configs/sam_mask.yaml + GUI 설정).",
    "preprocess_settings_saved": "전처리 설정이 저장되었습니다.",
    "preprocess_invalid_float": "{field}에 잘못된 숫자: {value}",
    "preprocess_already_running": "이미 전처리 단계가 실행 중입니다.",
    # ConfigTab
    "preset": "프리셋:",
    "save": "저장",
    "save_dirty_tooltip": "저장되지 않은 편집이 있습니다. Save를 누르면 variant 파일에 기록됩니다 (학습/전처리 시작 시 자동 저장됨).",
    "train": "학습",
    "test": "테스트",
    "stop": "정지",
    "log_placeholder": "학습 출력이 여기에 표시됩니다...",
    "from_base": "base.toml에서 상속",
    "saved": "저장 완료",
    "saved_file": "{name} 저장됨",
    "invalid_toml": "잘못된 TOML",
    "error": "오류",
    "accelerate_not_found": "PATH에서 accelerate를 찾을 수 없습니다",
    "preprocess": "전처리",
    "preprocess_required": "학습 전에 전처리를 먼저 실행해주세요.",
    "preprocess_existing_caches_title": "기존 캐시를 그대로 재사용합니다",
    "preprocess_existing_caches_body": (
        "다음 경로에 이미 캐시 파일이 있습니다:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "전처리는 기존 캐시를 그대로 재사용합니다 — 삭제하거나 다시 "
        "만들지 않습니다. 누락된 항목만 새로 처리됩니다.\n\n"
        "캡션을 수정했거나 토크나이저 설정 등을 바꿔서 캐시를 처음부터 "
        "다시 만들고 싶다면, 취소를 누르고 캐시 폴더를 직접 삭제한 뒤 "
        "다시 실행하세요."
    ),
    "preprocess_cache_count_latents": "VAE 잠재변수 {n}개 (.npz)",
    "preprocess_cache_count_te": "텍스트 임베딩 {n}개 (_te.safetensors)",
    "preprocess_cache_count_pe": "PE 피처 {n}개 (_pe.safetensors)",
    "train_using_cache_title": "기존 캐시 데이터셋으로 학습할까요?",
    "train_using_cache_body": (
        "다음 경로에 이미 전처리된 데이터셋 캐시가 있습니다:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "학습은 이 캐시를 그대로 재사용합니다. 새 이미지를 추가했거나 "
        "캡션을 수정해서 다시 반영하고 싶다면, 취소를 누르고 먼저 "
        "전처리를 실행하세요.\n\n"
        "기존 캐시로 학습을 진행할까요?"
    ),
    "train_autopreprocess_log": (
        "전처리 캐시가 없어 전처리를 먼저 실행한 뒤 자동으로 학습을 시작합니다.\n"
    ),
    "train_preprocessing": "전처리 중…",
    "no_lora_for_test": "테스트할 LoRA가 output/ckpt/에 없습니다. 먼저 학습을 실행하세요.",
    "test_output_title": "최신 테스트 출력",
    "test_output_empty": "output/tests/가 비어 있습니다.",
    "finished": "--- 완료 (종료 코드 {code}) ---",
    "starting": "시작 중… (torch / accelerate 로딩)",
    "update_success_title": "업데이트 완료",
    "update_success_message": (
        "anima_lora이(가) {v}(으)로 업데이트되었습니다.\n\n"
        "변경 사항을 적용하려면 GUI를 종료하고 다시 실행해 주세요."
    ),
    "update_success_badge": "{v}(으)로 업데이트됨 (재실행 필요)",
    "update_dryrun_done_title": "드라이런 완료",
    "update_dryrun_done_message": (
        "드라이런이 완료되었습니다. 실제 변경된 파일은 없습니다. "
        "어떤 변경이 일어날지 로그를 확인하세요."
    ),
    "update_failed_title": "업데이트 실패",
    "update_failed_message": (
        "업데이트가 코드 {code}(으)로 종료되었습니다. "
        "자세한 내용은 로그를 확인하세요. 작업 트리가 일부만 변경되었을 수 있습니다."
    ),
    "resume_checkpoint_title": "학습을 재개할까요?",
    "resume_checkpoint_question": (
        "재개 가능한 체크포인트가 감지되었습니다 (스텝 {step}).\n\n"
        "• 예 — 스텝 {step}부터 학습을 재개합니다\n"
        "• 아니오 — 기존 체크포인트를 삭제하고 처음부터 새로 학습합니다\n"
        "• 취소 — 학습을 시작하지 않습니다"
    ),
    "resume_checkpoint_delete_failed": "기존 체크포인트 상태를 삭제하지 못했습니다:\n{error}",
    "locked_by_preset": "프리셋에 의해 잠김 (이 VRAM 프로필의 성능 설정은 고정되어 있습니다)",
    "lora_variants": "LoRA 변형",
    "variant": "변형:",
    "apply_variant": "적용",
    "apply_variant_tooltip": "아래 폼을 이 variant의 프리셋 값으로 채웁니다. Save를 누르기 전까지는 디스크에 저장되지 않습니다.",
    "show_guide": "가이드",
    "show_guide_tooltip": "오른쪽 패널에 variant 가이드와 Apply 동작 설명을 표시합니다.",
    "click_field_for_help": "필드 라벨을 클릭하면 설명이 여기에 표시됩니다.",
    "no_help_available": "이 필드에 대한 설명이 없습니다.",
    "extra_args_toggle": "+ 추가 인자",
    "extra_args_placeholder": "폼에 없는 필드를 TOML 형식으로 입력. 예:\nmy_new_flag = true\nsome_value = 5e-5",
    "extra_args_tooltip": "폼에 없는 설정 키를 추가합니다. Save 시 TOML로 파싱되어 현재 variant 파일에 병합되며, 폼이 새로고침되어 위젯으로 표시됩니다. 동일 키가 폼에도 있는 경우 여기 입력값이 우선합니다.",
    "new_variant": "+ 새 Variant",
    "new_variant_tooltip": "configs/gui-methods/custom/<name>.toml에 새 커스텀 variant를 생성합니다.",
    "new_variant_prompt": "새 variant 이름 (configs/gui-methods/custom/<name>.toml에 저장됨).\n영문/숫자/_/- 만 사용 가능합니다.",
    "new_variant_invalid": "잘못된 이름. 영문, 숫자, _, - 만 사용 가능합니다.",
    "new_variant_exists": "Variant '{name}'이(가) 이미 존재합니다.",
    "basic_section": "기본 설정",
    "advanced_section": "고급 설정 (클릭하여 펼치기)",
    # AdapterTab (IP-Adapter / EasyControl)
    "adapter_source_dir": "소스 데이터셋:",
    "adapter_cache_dir": "캐시 디렉토리:",
    "adapter_n_pairs": "이미지 {n}개 / 캡션 {c}개 쌍",
    "adapter_n_caches": "캐시 {n}개",
    "adapter_preprocess": "전처리 (리사이즈 + VAE + 텍스트)",
    "adapter_preprocess_pe": "전처리 (리사이즈 + VAE + 텍스트 + PE)",
    "adapter_train": "학습",
    "adapter_stop": "정지",
    "adapter_log_placeholder": "실행 출력이 여기에 표시됩니다...",
    "adapter_no_dataset": "소스 데이터셋 디렉토리가 없습니다. 디렉토리를 만들고 이미지+캡션 쌍을 넣어주세요.",
    "adapter_open_dir": "디렉토리 열기",
    "n_images": "이미지 {n}개",
    # ImageViewerTab
    "directory": "디렉토리:",
    "dataset_reload": "새로고침",
    "dataset_reload_tooltip": "현재 디렉토리를 다시 스캔해서 이미지 목록과 선택을 갱신합니다.",
    "dataset_add_dir": "디렉토리 추가…",
    "dataset_add_dir_tooltip": "다른 디렉토리를 골라 이번 세션 동안 드롭다운에 추가합니다.",
    "dataset_add_dir_picker": "추가할 디렉토리 선택",
    "dataset_add_dir_already": "'{name}' 디렉토리는 이미 목록에 있습니다.",
    "dataset_search_placeholder": "파일 이름 검색…",
    "dataset_sort_asc_tooltip": "오름차순 정렬 (A→Z, 클릭하여 반전)",
    "dataset_sort_desc_tooltip": "내림차순 정렬 (Z→A, 클릭하여 반전)",
    "dataset_mask_overlay": "마스크 오버레이 표시",
    "dataset_view_list_tooltip": "리스트 뷰 (클릭하면 트리 뷰로 전환)",
    "dataset_view_tree_tooltip": "폴더 트리 뷰 (클릭하면 리스트 뷰로 전환)",
    "n_images_filtered": "{shown} / {total} 이미지",
    "caption": "캡션:",
    "no_caption": "(캡션 없음)",
    "caption_save": "저장",
    "caption_revert": "되돌리기",
    "caption_versions": "이력…",
    "caption_dirty_marker": " *",
    "caption_diff_stats": "(+{add} / −{rem})",
    "caption_diff_clean": "(변경 없음)",
    "caption_save_failed": "캡션 저장 실패: {err}",
    "caption_unsaved_title": "저장되지 않은 캡션",
    "caption_unsaved_body": "캡션 편집 사항이 저장되지 않았습니다. 전환하기 전에 저장할까요?",
    "caption_versions_title": "캡션 이력 — {name}",
    "caption_versions_empty": "(이전 버전 없음)",
    "caption_versions_restore": "선택 버전으로 되돌리기",
    "caption_versions_close": "닫기",
    "caption_no_history": "이 캡션에는 아직 이력이 없습니다.",
    "caption_guideline_html": (
        "<b>순서:</b> 등급 → 인원수 → 캐릭터 (작품) → 작품 → "
        "<span style='color:#c9a227;'>@작가</span> → 내용 태그. "
        "영역별 하위 섹션: 직전 태그를 <code>.</code> 으로 끝낸 뒤 "
        "<span style='color:#5e8eb0;'>On the&nbsp;…,</span> 또는 "
        "<span style='color:#5e8eb0;'>In the&nbsp;…,</span> 로 시작. "
        "첫 <code>@작가</code> 태그까지는 순서가 고정되고, 그 이후는 "
        "섹션 내에서 셔플됩니다. "
        "<b>작가 정보가 없을 때:</b> "
        "<span style='color:#c9a227;'>@no-artist</span> 를 자리표시자로 "
        "넣어주세요 — 셔플 경계 역할만 하고 토큰화 직전에 제거되어 "
        "모델까지 전달되지 않습니다."
    ),
    # Language
    "language": "언어:",
    # Guidebook
    "guidebook": "📖 가이드북",
    "guidebook_tooltip": "한국어 종합 가이드 열기 (docs/guidelines/가이드북.md)",
    "guidebook_missing": "가이드를 찾을 수 없습니다: {path}",
    "guidebook_open_external": "시스템 뷰어로 열기",
    "guidebook_close": "닫기",
    # Top-bar buttons (models / update / report issue)
    "models_btn": "모델",
    "models_btn_tooltip": "모델 체크포인트 다운로드 / 재다운로드 (Anima 베이스, SAM3, MIT, IP-Adapter 인코더)",
    "update_btn": "업데이트",
    "update_btn_tooltip": "GitHub에서 최신 anima_lora 릴리스를 가져오고 uv sync를 실행합니다",
    "report_issue": "이슈 신고",
    "report_issue_tooltip": "브라우저에서 GitHub 이슈 트래커 열기",
    "experimental_features": "🧪 실험 기능",
    "experimental_features_tooltip": "Postfix 및 IP-Adapter / EasyControl 탭 열기 (이미지 조건부 방식)",
    "experimental_features_title": "실험 기능",
    # Models dialog
    "models_title": "모델 다운로드",
    "models_intro": "아래에서 모델 그룹을 선택하거나 '전체 다운로드'로 표준 세트 "
    "(Anima + SAM3 + MIT + PE)를 받으세요. 파일은 models/ 아래에 저장됩니다.",
    "models_download_all": "전체 다운로드 (Anima + SAM3 + MIT + PE)",
    "models_download": "다운로드",
    "models_redownload": "재다운로드",
    "models_installed": "✓ 설치됨",
    "models_missing": "✗ 없음",
    "model_anima": "Anima — DiT + 텍스트 인코더 + VAE",
    "model_sam3": "SAM3 — 말풍선 마스킹",
    "model_mit": "MIT — 만화 텍스트 마스킹",
    "model_pe": "PE-Core-L14-336 — IP-Adapter 비전 인코더",
    # Update dialog
    "update_title": "anima_lora 업데이트",
    "update_warning": "업데이트는 GitHub에서 최신 릴리스를 받아 작업 트리를 덮어씁니다 "
    "(datasets, output/, models/는 보존됩니다). configs/methods/와 "
    "configs/gui-methods/에 직접 수정한 내용은, 그대로 유지할지 또는 "
    "최신 버전으로 덮어쓸지(기존 파일은 자동 백업됨) 선택하세요. "
    "먼저 'Dry run'으로 변경사항을 미리 확인할 수 있습니다.",
    "update_dry_run": "Dry run",
    "update_run": "업데이트 실행",
    "update_run_keep": "업데이트 — 내 설정 유지",
    "update_run_overwrite": "업데이트 — 설정 덮어쓰기 (기존 백업)",
    "update_confirm": "anima_lora 소스 파일이 다시 작성됩니다. 계속하시겠습니까?",
    "update_check_now": "업데이트 확인",
    "update_view_release": "GitHub에서 보기",
    "update_current_version": "현재: {v}",
    "update_latest_version": "최신: {v}",
    "update_no_baseline": "알 수 없음 (manifest 없음)",
    "update_status_checking": "확인 중…",
    "update_status_uptodate": "✓ 최신 버전입니다",
    "update_status_available": "● 업데이트 있음",
    "update_status_unknown": "? 비교 불가 (로컬 manifest 없음)",
    "update_status_failed": "✗ 확인 실패",
    "update_release_notes": "릴리스 노트:",
    "update_no_release_notes": "(릴리스 설명이 없습니다)",
    "update_check_error": "GitHub에 접속할 수 없습니다: {err}",
    # MergeTab
    "n_files": "파일 {n}개",
    "merge_no_adapter": "어댑터를 찾을 수 없습니다",
    "merge_no_adapter_msg": "어댑터가 선택되지 않았거나 파일이 존재하지 않습니다.",
    "merge_no_selection": "목록에서 체크포인트를 선택하여 스캔하세요.",
    "merge_verdict_ready": "✓ 병합 준비됨",
    "merge_verdict_partial": "△ 부분 병합 — LoRA는 병합되고 ReFT는 제외됩니다",
    "merge_verdict_hydra": "✗ HydraLoRA moe — 레이어 로컬 라우터는 병합할 수 없습니다",
    "merge_verdict_postfix_only": "✗ postfix/prefix 전용 — 가중치 델타가 아닙니다",
    "merge_verdict_reft_only": "✗ ReFT 전용 — 블록 후크만 있고 병합할 LoRA가 없습니다",
    "merge_verdict_unknown": "? 인식되는 어댑터 키가 없습니다",
    "merge_options": "병합 옵션",
    "merge_base_dit": "베이스 DiT:",
    "merge_multiplier": "강도 배수:",
    "merge_multiplier_tip": "병합 시 적용할 LoRA 강도 (1.0 = 전체 강도)",
    "merge_dtype": "저장 dtype:",
    "merge_out": "출력:",
    "merge_out_placeholder": "(자동: <adapter>_merged.safetensors)",
    "merge_allow_partial": "부분 병합 허용 (ReFT / Hydra / postfix 키 제외)",
    "merge_allow_partial_tip": "병합 불가능한 컴포넌트가 있어도 진행합니다. 제외된 컴포넌트는 병합된 DiT에 반영되지 않습니다.",
    "merge_button": "DiT에 병합",
    "merge_log_placeholder": "병합 출력이 여기에 표시됩니다...",
    "merge_pick_dir": "어댑터 디렉토리 선택",
    "merge_pick_file": "어댑터 .safetensors 선택",
    "merge_pick_dit": "베이스 DiT .safetensors 선택",
    "merge_pick_out": "병합된 DiT 저장 위치...",
    "browse": "찾아보기…",
}
