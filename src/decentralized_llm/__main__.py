import os
import time

os.environ["DECENTRALIZED_WORKER_STARTED"] = str(time.time())

from decentralized_llm.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
