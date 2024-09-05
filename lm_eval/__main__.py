import argparse
import json
import logging
import os
import sys
from functools import partial
from typing import Union

from lm_eval import evaluator, utils
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string


def _int_or_none_list_arg_type(
    min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","
):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'"
        )
    elif num_items != max_len:
        logging.warning(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'. "
            "Missing values will be filled with defaults."
        )
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(
            default_items[num_items:]
        )  # extend items list with missing defaults

    return items


def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(
                    f"Argument '{action.dest}' doesn't have a type specified."
                )
            else:
                continue


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        type=str,
        metavar="task1,task2",
        help="Comma-separated list of task names or task groupings to evaluate on.\nTo get full list of tasks, use one of the commands `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        type=str,
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        type=str,
        nargs="?",
        const=True,
        default=False,
        help=(
            "If True, apply chat template to the prompt. "
            "Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt. "
            "To apply a specific template from the available list of templates, provide the template name as an argument. "
            "E.g. `--apply_chat_template template_name`"
        ),
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        type=str,
        default=None,
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`."
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    parser.add_argument(
        "--wandb_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    return parser


def parse_eval_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    check_argument_types(parser)
    return parser.parse_args()


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None: # args 인자는 argparse.Namespace 또는 None
    '''
    1. 명령행 인자 파싱
    2. 로깅 설정
    3. 평가 트래커 설정
    4. few-shot 및 기타 설정 확인
    5. 작업 로드
    6. 모델 평가
    7. 결과 저장 및 로깅
    '''
    if not args: # 명령행 인자 파싱, 없으면 setup_parser와 parse_eval_args로 파싱
        # we allow for args to be passed externally, else we parse them ourselves
        parser = setup_parser()
        args = parse_eval_args(parser)

    if args.wandb_args:
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args)) # 로깅 설정 초기화

    eval_logger = utils.eval_logger # 로깅 설정 초기화
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}")) # 로깅 레벨 설정
    eval_logger.info(f"Verbosity set to {args.verbosity}") # 로깅 레벨 출력
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # 병렬 토크나이저 비활성화

    # 평가 트래커 설정 update the evaluation tracker args with the output path and the HF token
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}" # 토큰 설정
    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args) # EvaluationTracker 객체 생성

    # Few-shot 및 기타 설정 확인하고 필요한 경우 예외 발생
    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError(
            "Specify --output_path if providing --log_samples or --predict_only"
        )

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError(
            "When `fewshot_as_multiturn` is selected, `apply_chat_template` must be set (either to `True` or to the chosen template name)."
        )

    if (
        args.num_fewshot is None or args.num_fewshot == 0
    ) and args.fewshot_as_multiturn:
        raise ValueError(
            "If fewshot_as_multiturn is set, num_fewshot must be greater than 0."
        )

    # 작업 로드 
    # 평가할 작업 로드하고 작업 목록 확인/ 작업이 지정되지 않았거나 잘못된 작업이 지정된 경우 예외 발생 시킴
    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning(
            "Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub."
        )

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    # 실제 평가할 작업 목록 확인 => task names
    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        print(task_manager.list_all_tasks())
        sys.exit()
    elif args.tasks == "list_groups":
        print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        print(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks): # 지정한 task가 디렉토리에 있을 경우
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else: # 지정한 task가 디렉토리에 없을 경우
            task_list = args.tasks.split(",") # 쉼표로 구분된 두 개 이상의 task 목록일 경우 배열로 변환
            task_names = task_manager.match_tasks(task_list) # task_manager 객체의 match_tasks 메서드로 task 목록 일치 여부 확인
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task): # task가 파일일 경우
                    config = utils.load_yaml_config(task) # yaml 파일 로드 (task 설정)
                    task_names.append(config) # task_names에 task 설정 추가
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task # task_list에 있는 task 중 task_names에 없는 task 목록
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing) # task_missing 배열을 쉼표로 구분된 문자열로 변환
                eval_logger.error(
                    f"Tasks were not found: {missing}\n" # 찾을 수 없는 task 출력
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    # Respect user's value passed in via CLI, otherwise default to True and add to comma-separated model args
    if args.trust_remote_code: #원격저장소에서 모델을 다운로드하거나 데이터를 신뢰하는지 설정
        eval_logger.info(
            "Passed `--trust_remote_code`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`"
        )
        # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
        # because it's already been determined based on the prior env var before launching our
        # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
        import datasets # 데이터셋 임포트

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True # HF 데이터셋 신뢰 설정

        args.model_args = args.model_args + ",trust_remote_code=True" # args.model_args 문자열에 trust_remote_code=True 추가

    eval_logger.info(f"Selected Tasks: {task_names}")

    # 요청 캐싱 인자를 딕셔너리로 변환
    request_caching_args = request_caching_arg_to_dict(
        cache_requests=args.cache_requests 
    )

    results = evaluator.simple_evaluate( # 모델 평가 및 results 변수에 평가 결과를 저장
        model=args.model, # 모델 객체
        model_args=args.model_args, # 모델 인자
        tasks=task_names,
        num_fewshot=args.num_fewshot, # few-shot 학습 샘플 수
        batch_size=args.batch_size, 
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity, # 데이터 무결성 검사 수행할지 여부
        write_out=args.write_out, # 평가 결과를 파일에 쓸지 여부
        log_samples=args.log_samples, # 샘플을 로그로 남길지 여부
        evaluation_tracker=evaluation_tracker, # 평가 과정을 추적할 객체
        system_instruction=args.system_instruction, # 시스템 지시 사항
        apply_chat_template=args.apply_chat_template, # 채팅 템플릿 적용 여부
        fewshot_as_multiturn=args.fewshot_as_multiturn, # few-shot을 멀티턴 대화로 사용할지 여부
        gen_kwargs=args.gen_kwargs, # 생성 함수에 전달한 추가 인자
        task_manager=task_manager, # 작업을 관리할 객체
        verbosity=args.verbosity, # 로그의 상세 수준
        predict_only=args.predict_only, # 예측만 수행할지 여부
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        **request_caching_args, # 요청 캐싱과 관련된 추가 인자
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples") # 샘플 추출
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        # Add W&B logging
        if args.wandb_args:
            try:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if args.log_samples:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:
                eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if args.log_samples else None
        )

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        if (
            evaluation_tracker.push_results_to_hub
            or evaluation_tracker.push_samples_to_hub
        ):
            evaluation_tracker.recreate_metadata_card()

        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if args.wandb_args:
            # Tear down wandb run once all the logging is done.
            wandb_logger.run.finish()
        
        ''' test: what is sample and results '''
        print(f'results 출력합니다 : {results}')
        print(f'sample 출력합니다 : {samples}')


if __name__ == "__main__":
    cli_evaluate()
